from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def build_adjacency(
    trailing_returns: np.ndarray,
    sectors: Sequence[str],
    mode: str,
) -> np.ndarray:
    """Build a history-only sector/correlation graph with explicit self-loops."""
    n_assets = len(sectors)
    sector = np.equal.outer(np.asarray(sectors, dtype=object), np.asarray(sectors, dtype=object)).astype(np.float32)
    np.fill_diagonal(sector, 1.0)

    if trailing_returns.shape[0] < 2:
        correlation = np.eye(n_assets, dtype=np.float32)
    else:
        correlation = np.corrcoef(trailing_returns, rowvar=False)
        correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
        correlation = np.maximum(correlation, 0.0).astype(np.float32)
        np.fill_diagonal(correlation, 1.0)

    if mode == "sector":
        adjacency = sector
    elif mode == "correlation":
        adjacency = correlation
    elif mode == "combined":
        adjacency = 0.5 * sector + 0.5 * correlation
        np.fill_diagonal(adjacency, 1.0)
    else:
        raise ValueError(f"Unknown graph mode: {mode}")
    if not np.isfinite(adjacency).all() or not np.allclose(adjacency, adjacency.T, atol=1e-7):
        raise RuntimeError("Adjacency matrix is not finite and symmetric")
    return adjacency.astype(np.float32)


def select_period_end_dates(dates: Iterable[pd.Timestamp], frequency: str) -> list[pd.Timestamp]:
    index = pd.DatetimeIndex(sorted(pd.to_datetime(list(dates)).unique()))
    if frequency == "daily":
        return list(index)
    if frequency == "weekly":
        periods = index.to_period("W-FRI")
    elif frequency == "monthly":
        periods = index.to_period("M")
    elif frequency == "quarterly":
        periods = index.to_period("Q")
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")
    frame = pd.DataFrame({"Date": index, "Period": periods})
    period_ends = list(frame.groupby("Period", sort=True)["Date"].max())
    # Enter at the first eligible close so every fold covers its stated start;
    # subsequent decisions use actual calendar period ends.
    return list(pd.DatetimeIndex([index[0], *period_ends]).unique())


class PanelWindowDataset(Dataset):
    """Dense, asset-agnostic windows that reject incomplete samples explicitly."""

    def __init__(
        self,
        panel: pd.DataFrame,
        symbols: Sequence[str],
        feature_columns: Sequence[str],
        macro_columns: Sequence[str],
        target_columns: Sequence[str],
        start: str,
        end: str,
        window_size: int,
        graph_lookback: int,
        graph_mode: str = "combined",
        require_targets: bool = True,
    ) -> None:
        self.symbols = list(symbols)
        self.feature_columns = list(feature_columns)
        self.macro_columns = list(macro_columns)
        self.target_columns = list(target_columns)
        self.target_date_columns = [
            column.replace("ForwardReturn", "ForwardDate") for column in self.target_columns
        ]
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.window_size = int(window_size)
        self.graph_lookback = int(graph_lookback)
        self.graph_mode = graph_mode
        self.require_targets = require_targets

        subset = panel[panel["Symbol"].isin(self.symbols)].copy()
        available = set(subset["Symbol"].unique())
        absent = [symbol for symbol in self.symbols if symbol not in available]
        if absent:
            raise ValueError(f"Panel lacks requested symbols: {absent}")
        needed = set(
            self.feature_columns
            + self.macro_columns
            + self.target_columns
            + (self.target_date_columns if require_targets else [])
            + ["Close", "Sector"]
        )
        missing_columns = needed - set(subset.columns)
        if missing_columns:
            raise ValueError(f"Panel lacks dataset columns: {sorted(missing_columns)}")

        self.dates = pd.DatetimeIndex(sorted(subset["Date"].unique()))
        self.date_to_position = {date: i for i, date in enumerate(self.dates)}
        self.sectors = [
            str(subset.loc[subset["Symbol"] == symbol, "Sector"].dropna().iloc[0]) for symbol in self.symbols
        ]
        self.feature_array = self._pivot_columns(subset, self.feature_columns)
        self.close_array = self._pivot_columns(subset, ["Close"])[:, :, 0]
        self.return_array = np.vstack(
            [np.full((1, len(self.symbols)), np.nan), np.diff(np.log(self.close_array), axis=0)]
        )
        self.macro_array = self._macro_columns(subset, self.macro_columns)
        self.target_array = self._pivot_columns(subset, self.target_columns) if self.target_columns else None
        self.target_date_array = (
            self._pivot_date_columns(subset, self.target_date_columns)
            if require_targets and self.target_date_columns
            else None
        )

        start_position = max(self.window_size - 1, self.graph_lookback)
        self.positions: list[int] = []
        self.rejected_positions: dict[str, int] = {
            "feature": 0,
            "target": 0,
            "target_boundary": 0,
            "close": 0,
        }
        for position in range(start_position, len(self.dates)):
            date = self.dates[position]
            if date < self.start or date > self.end:
                continue
            feature_window = self.feature_array[position - self.window_size + 1 : position + 1]
            if not np.isfinite(feature_window).all() or not np.isfinite(self.macro_array[position - self.window_size + 1 : position + 1]).all():
                self.rejected_positions["feature"] += 1
                continue
            if not np.isfinite(self.close_array[position]).all():
                self.rejected_positions["close"] += 1
                continue
            if require_targets and self.target_array is not None and not np.isfinite(self.target_array[position]).all():
                self.rejected_positions["target"] += 1
                continue
            if require_targets and self.target_date_array is not None:
                target_dates = self.target_date_array[position]
                if np.isnat(target_dates).any() or (target_dates > np.datetime64(self.end)).any():
                    self.rejected_positions["target_boundary"] += 1
                    continue
            self.positions.append(position)
        if not self.positions:
            raise RuntimeError(
                f"No complete windows for {self.start.date()}..{self.end.date()}; rejected={self.rejected_positions}"
            )

    def _pivot_columns(self, subset: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
        arrays = []
        for column in columns:
            pivot = subset.pivot(index="Date", columns="Symbol", values=column).reindex(
                index=self.dates, columns=self.symbols
            )
            arrays.append(pivot.to_numpy(dtype=np.float64))
        return np.stack(arrays, axis=-1)

    def _macro_columns(self, subset: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
        by_date = subset.sort_values(["Date", "Symbol"]).drop_duplicates("Date").set_index("Date")
        return by_date.reindex(self.dates)[list(columns)].to_numpy(dtype=np.float64)

    def _pivot_date_columns(self, subset: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
        arrays = []
        for column in columns:
            source = subset[["Date", "Symbol", column]].copy()
            source[column] = pd.to_datetime(source[column])
            pivot = source.pivot(index="Date", columns="Symbol", values=column).reindex(
                index=self.dates, columns=self.symbols
            )
            arrays.append(pivot.to_numpy(dtype="datetime64[ns]"))
        return np.stack(arrays, axis=-1)

    @lru_cache(maxsize=2048)
    def _adjacency(self, position: int) -> np.ndarray:
        trailing = self.return_array[position - self.graph_lookback + 1 : position + 1]
        return build_adjacency(trailing, self.sectors, self.graph_mode)

    def _state_arrays(self, position: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw_prices = self.feature_array[position - self.window_size + 1 : position + 1].copy()
        means = raw_prices.mean(axis=(0, 1), keepdims=True)
        standard_deviations = raw_prices.std(axis=(0, 1), keepdims=True)
        prices = (raw_prices - means) / np.where(standard_deviations > 1e-8, standard_deviations, 1.0)

        raw_macro = self.macro_array[position - self.window_size + 1 : position + 1].copy()
        macro_mean = raw_macro.mean(axis=0, keepdims=True)
        macro_std = raw_macro.std(axis=0, keepdims=True)
        macro = (raw_macro - macro_mean) / np.where(macro_std > 1e-8, macro_std, 1.0)
        prices = np.transpose(prices, (1, 0, 2)).astype(np.float32)
        macro = np.broadcast_to(macro[None, :, :], (len(self.symbols), *macro.shape)).astype(np.float32).copy()
        features = np.concatenate([prices, macro], axis=-1)
        return prices, macro, features

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        position = self.positions[index]
        prices, macro, features = self._state_arrays(position)
        labels = (
            self.target_array[position].astype(np.float32)
            if self.target_array is not None
            else np.empty((len(self.symbols), 0), dtype=np.float32)
        )
        return {
            "prices": torch.from_numpy(prices),
            "macro": torch.from_numpy(macro),
            "features": torch.from_numpy(features),
            "adj_matrix": torch.from_numpy(self._adjacency(position).copy()),
            "active_mask": torch.ones(len(self.symbols), dtype=torch.bool),
            "labels": torch.from_numpy(labels),
            "date": self.dates[position].strftime("%Y-%m-%d"),
        }

    def index_for_date(self, date: str | pd.Timestamp) -> int:
        target = pd.Timestamp(date)
        for index, position in enumerate(self.positions):
            if self.dates[position] == target:
                return index
        raise KeyError(f"Date is not an eligible sample: {target.date()}")

    def available_dates(self) -> list[pd.Timestamp]:
        return [self.dates[position] for position in self.positions]

    def decision_dates(self, frequency: str) -> list[pd.Timestamp]:
        return select_period_end_dates(self.available_dates(), frequency)

    def sample_for_date(self, date: str | pd.Timestamp) -> Dict[str, Any]:
        return self[self.index_for_date(date)]

    def close_on(self, date: str | pd.Timestamp) -> np.ndarray:
        position = self.date_to_position[pd.Timestamp(date)]
        values = self.close_array[position]
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite close prices at {pd.Timestamp(date).date()}")
        return values.copy()

    def raw_state_summary(self, date: str | pd.Timestamp) -> np.ndarray:
        position = self.date_to_position[pd.Timestamp(date)]
        raw = self.feature_array[position - self.window_size + 1 : position + 1]
        last = raw[-1]
        summary = np.concatenate(
            [
                np.mean(last, axis=0),
                np.std(last, axis=0),
                np.min(last, axis=0),
                np.max(last, axis=0),
                self.macro_array[position],
            ]
        )
        if not np.isfinite(summary).all():
            raise RuntimeError(f"Non-finite alpha state summary at {pd.Timestamp(date).date()}")
        return summary.astype(np.float32)
