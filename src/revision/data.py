from __future__ import annotations

import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import requests

from .utils import atomic_write_json, git_commit, package_versions, sha256_bytes, sha256_file, stable_order


PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
FACTOR_COLUMNS = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF"]
YAHOO_FINANCE_URL = "https://finance.yahoo.com/"
FAMA_FRENCH_DATASET = "F-F_Research_Data_5_Factors_2x3_daily"
FAMA_FRENCH_URL = (
    "http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)


def data_configuration_hash(config: Dict[str, Any]) -> str:
    payload = {
        "data": config["data"],
        "folds": config["folds"],
        "prepared_dir": config["paths"]["prepared_dir"],
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(serialized.encode("utf-8"))


def _canonical_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace("-", ".")


def _provider_symbol(symbol: str) -> str:
    return symbol.replace(".", "-")


def _deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def fetch_membership_snapshot(config: Dict[str, Any], prepared_dir: Path) -> pd.DataFrame:
    spec = config["data"]["membership"]
    response = requests.get(
        spec["url"],
        timeout=60,
        headers={"User-Agent": "sms-tgnn-ddpg-repro/1.0 (academic reproducibility)"},
    )
    response.raise_for_status()
    html_path = prepared_dir / f"sp500_membership_{spec['as_of']}.html"
    html_path.write_bytes(response.content)

    tables = pd.read_html(io.StringIO(response.text))
    matches = [table for table in tables if {"Symbol", "GICS Sector"}.issubset(table.columns)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one S&P 500 constituent table, found {len(matches)}")
    membership = matches[0].rename(columns={"GICS Sector": "Sector"})[["Symbol", "Sector"]].copy()
    membership["Symbol"] = membership["Symbol"].map(_canonical_symbol)
    membership["Sector"] = membership["Sector"].astype(str).str.strip()
    if membership["Symbol"].duplicated().any():
        raise RuntimeError("Membership snapshot contains duplicate symbols")
    membership.attrs["html_path"] = str(html_path)
    membership.attrs["html_sha256"] = sha256_file(html_path)
    membership.attrs["acquired_at_utc"] = datetime.now(timezone.utc).isoformat()
    return membership


def _extract_downloaded_symbol(downloaded: pd.DataFrame, provider: str) -> pd.DataFrame:
    if not isinstance(downloaded.columns, pd.MultiIndex):
        return downloaded.copy()
    level0 = set(map(str, downloaded.columns.get_level_values(0)))
    level1 = set(map(str, downloaded.columns.get_level_values(1)))
    if provider in level0:
        return downloaded[provider].copy()
    if provider in level1:
        return downloaded.xs(provider, axis=1, level=1).copy()
    return pd.DataFrame(index=downloaded.index)


def download_yfinance_prices(
    symbols: Iterable[str],
    start: str,
    end_exclusive: str,
    chunk_size: int = 50,
    cache_dir: str | Path | None = None,
    retries: int = 3,
) -> pd.DataFrame:
    import yfinance as yf

    canonical = list(dict.fromkeys(_canonical_symbol(symbol) for symbol in symbols))
    rows: list[pd.DataFrame] = []
    cache_root = Path(cache_dir) if cache_dir is not None else None
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(canonical), chunk_size):
        chunk = canonical[offset : offset + chunk_size]
        providers = [_provider_symbol(symbol) for symbol in chunk]
        cache_path = cache_root / f"chunk_{offset:04d}.csv.gz" if cache_root is not None else None
        pending = dict(zip(chunk, providers))
        chunk_rows: list[pd.DataFrame] = []
        if cache_path is not None and cache_path.exists():
            cached = pd.read_csv(cache_path, parse_dates=["Date"])
            if set(cached["Symbol"].unique()).issubset(set(chunk)):
                # The query window is part of the cache path. Reuse the frozen
                # first successful snapshot, including explicitly absent or
                # delisted symbols, instead of allowing later provider drift.
                rows.append(cached)
                continue
        for attempt in range(retries):
            if not pending:
                break
            downloaded = yf.download(
                list(pending.values()),
                start=start,
                end=end_exclusive,
                auto_adjust=False,
                actions=False,
                group_by="ticker",
                threads=True,
                progress=False,
                timeout=30,
            )
            completed: list[str] = []
            for symbol, provider in pending.items():
                one = _extract_downloaded_symbol(downloaded, provider)
                if one.empty or "Close" not in one:
                    continue
                one = one.rename_axis("Date").reset_index()
                one["Date"] = pd.to_datetime(one["Date"]).dt.tz_localize(None)
                close = pd.to_numeric(one["Close"], errors="coerce")
                adjusted = pd.to_numeric(one.get("Adj Close", close), errors="coerce")
                ratio = adjusted.div(close.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
                for column in ("Open", "High", "Low", "Close"):
                    one[column] = pd.to_numeric(one[column], errors="coerce") * ratio
                one["Volume"] = pd.to_numeric(one.get("Volume"), errors="coerce")
                one["Symbol"] = symbol
                chunk_rows.append(one[["Date", "Symbol", *PRICE_COLUMNS]])
                completed.append(symbol)
            for symbol in completed:
                pending.pop(symbol, None)
            if pending and attempt + 1 < retries:
                time.sleep(2**attempt)
        if chunk_rows:
            chunk_frame = pd.concat(chunk_rows, ignore_index=True)
            chunk_frame = chunk_frame.sort_values(["Symbol", "Date"]).drop_duplicates(["Date", "Symbol"], keep="last")
            if cache_path is not None:
                _deterministic_gzip_csv(chunk_frame, cache_path)
            rows.append(chunk_frame)
    if not rows:
        raise RuntimeError("Yahoo Finance returned no usable price histories")
    prices = pd.concat(rows, ignore_index=True)
    prices = prices.dropna(subset=["Date", "Symbol", "Close"])
    prices = prices.sort_values(["Symbol", "Date"]).drop_duplicates(["Date", "Symbol"], keep="last")
    return prices.reset_index(drop=True)


def load_csv_prices(path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path, parse_dates=["Date"])
    missing = {"Date", "Symbol", "Sector", *PRICE_COLUMNS} - set(frame.columns)
    if missing:
        raise ValueError(f"Raw CSV is missing columns: {sorted(missing)}")
    frame["Symbol"] = frame["Symbol"].map(_canonical_symbol)
    factors = frame[["Date", *[column for column in FACTOR_COLUMNS if column in frame]]].drop_duplicates("Date")
    absent_factors = set(FACTOR_COLUMNS) - set(factors.columns)
    if absent_factors:
        raise ValueError(f"Fixture CSV is missing factor columns: {sorted(absent_factors)}")
    return frame[["Date", "Symbol", "Sector", *PRICE_COLUMNS]].copy(), factors


def download_fama_french(start: str, end: str) -> pd.DataFrame:
    import pandas_datareader.data as web

    dataset = web.DataReader(
        FAMA_FRENCH_DATASET, "famafrench", start=start, end=end
    )
    factors = dataset[0].copy() / 100.0
    if isinstance(factors.index, pd.PeriodIndex):
        factors.index = factors.index.to_timestamp()
    else:
        factors.index = pd.to_datetime(factors.index)
    factors.index = factors.index.tz_localize(None)
    factors.columns = [str(column).replace("-", "_").replace(" ", "") for column in factors.columns]
    factors.index.name = "Date"
    factors = factors.reset_index()
    if set(FACTOR_COLUMNS) - set(factors.columns):
        raise RuntimeError(f"Fama-French source lacks required columns: {FACTOR_COLUMNS}")
    return factors[["Date", *FACTOR_COLUMNS]].sort_values("Date").reset_index(drop=True)


def _coverage(prices: pd.DataFrame, calendar: pd.DatetimeIndex, start: str, end: str) -> Dict[str, float]:
    expected = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    if len(expected) == 0:
        raise ValueError("Coverage calendar is empty")
    observed = prices.loc[
        prices["Date"].between(pd.Timestamp(start), pd.Timestamp(end)) & prices["Close"].notna()
    ].groupby("Symbol")["Date"].nunique()
    return {str(symbol): float(count / len(expected)) for symbol, count in observed.items()}


def select_universes(
    config: Dict[str, Any], membership: pd.DataFrame, prices: pd.DataFrame, calendar: pd.DatetimeIndex
) -> Dict[str, Any]:
    spec = config["data"]["universe"]
    minimum = float(spec["minimum_coverage"])
    primary_fold = config["folds"]["primary"]
    train_coverage = _coverage(
        prices, calendar, config["data"]["source_start"], primary_fold["validation"]["end"]
    )
    reserve_coverage = _coverage(
        prices, calendar, "2020-01-02", primary_fold["test"]["end"]
    )
    sector_by_symbol = dict(zip(membership["Symbol"], membership["Sector"]))
    exact_test = [_canonical_symbol(symbol) for symbol in spec["exact_test_symbols"]]
    missing_test = [symbol for symbol in exact_test if reserve_coverage.get(symbol, 0.0) < minimum]
    if missing_test:
        raise RuntimeError(f"Exact paper test symbols fail the coverage gate: {missing_test}")

    reserve_candidates = [
        symbol
        for symbol in membership["Symbol"]
        if symbol not in exact_test and reserve_coverage.get(symbol, 0.0) >= minimum
    ]
    reserve: list[str] = []
    reserve_sectors: set[str] = set()
    for symbol in stable_order(reserve_candidates, spec["selection_salt"] + "|reserve"):
        sector = sector_by_symbol[symbol]
        if sector in reserve_sectors:
            continue
        reserve.append(symbol)
        reserve_sectors.add(sector)
        if len(reserve) == int(spec["reserve_size"]):
            break
    if len(reserve) != int(spec["reserve_size"]):
        raise RuntimeError("Not enough sector-distinct reserve symbols passed the coverage gate")

    excluded = set(exact_test) | set(reserve)
    train: list[str] = []
    per_sector = int(spec["per_sector_train"])
    sectors = sorted(membership["Sector"].unique())
    for sector in sectors:
        candidates = [
            symbol
            for symbol in membership.loc[membership["Sector"] == sector, "Symbol"]
            if symbol not in excluded and train_coverage.get(symbol, 0.0) >= minimum
        ]
        selected = stable_order(candidates, spec["selection_salt"] + "|train")[:per_sector]
        if len(selected) != per_sector:
            raise RuntimeError(f"Sector {sector!r} has only {len(selected)} eligible training symbols")
        train.extend(selected)

    n5 = exact_test[:5]
    selection_status: dict[str, str] = {}
    for symbol in membership["Symbol"]:
        if symbol in exact_test:
            status = "selected_exact_test"
        elif symbol in reserve:
            status = "selected_reserve"
        elif symbol in train:
            status = "selected_train"
        elif train_coverage.get(symbol, 0.0) < minimum:
            status = "excluded_train_coverage"
        else:
            status = "eligible_not_selected"
        selection_status[symbol] = status

    return {
        "train": train,
        "test": exact_test,
        "reserve": reserve,
        "n5": n5,
        "n10": exact_test,
        "n15": exact_test + reserve,
        "sector_by_symbol": {
            symbol: sector_by_symbol[symbol] for symbol in sorted(set(train + exact_test + reserve))
        },
        "coverage": {
            symbol: {
                "train_validation": train_coverage.get(symbol, 0.0),
                "test": reserve_coverage.get(symbol, 0.0),
            }
            for symbol in membership["Symbol"]
        },
        "selection_status": selection_status,
    }


def configured_universes(config: Dict[str, Any], prices: pd.DataFrame) -> Dict[str, Any]:
    spec = config["data"]["universe"]
    train = [_canonical_symbol(symbol) for symbol in spec["configured_train_symbols"]]
    test = [_canonical_symbol(symbol) for symbol in spec["exact_test_symbols"]]
    reserve = [_canonical_symbol(symbol) for symbol in spec.get("configured_reserve_symbols", [])]
    sector_by_symbol = prices.drop_duplicates("Symbol").set_index("Symbol")["Sector"].to_dict()
    unknown = [symbol for symbol in set(train + test + reserve) if symbol not in sector_by_symbol]
    if unknown:
        raise ValueError(f"Configured symbols are absent from the CSV: {unknown}")
    return {
        "train": train,
        "test": test,
        "reserve": reserve,
        "n5": test[: min(5, len(test))],
        "n10": test,
        "n15": test + reserve,
        "sector_by_symbol": {
            symbol: sector_by_symbol[symbol] for symbol in sorted(set(train + test + reserve))
        },
        "coverage": {},
    }


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    relative_strength = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + relative_strength)
    rsi = rsi.mask((loss == 0.0) & (gain > 0.0), 100.0)
    rsi = rsi.mask((gain == 0.0) & (loss > 0.0), 0.0)
    rsi = rsi.mask((gain == 0.0) & (loss == 0.0), 50.0)
    return rsi


def engineer_features(prices: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol, source in prices.groupby("Symbol", sort=True):
        one = source.sort_values("Date").copy()
        close = one["Close"].astype(float)
        one["Ret1D"] = np.log(close).diff()
        previous_close = close.shift(1)
        one["OpenRel"] = one["Open"].div(previous_close) - 1.0
        one["HighRel"] = one["High"].div(close) - 1.0
        one["LowRel"] = one["Low"].div(close) - 1.0
        one["CloseRel20"] = close.div(close.rolling(20, min_periods=20).mean()) - 1.0
        volume = np.log1p(one["Volume"].astype(float))
        one["VolumeZ20"] = (volume - volume.rolling(20, min_periods=20).mean()).div(
            volume.rolling(20, min_periods=20).std(ddof=0).replace(0.0, np.nan)
        )
        for horizon in (5, 21, 63, 252):
            one[f"Momentum{horizon}"] = close.pct_change(horizon, fill_method=None)
        one["Volatility20"] = close.pct_change(fill_method=None).rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
        one["RSI14"] = _rsi(close)
        fast = close.ewm(span=12, adjust=False).mean()
        slow = close.ewm(span=26, adjust=False).mean()
        macd = fast - slow
        one["MACDHist"] = macd - macd.ewm(span=9, adjust=False).mean()
        for horizon in horizons:
            one[f"ForwardReturn{horizon}D"] = close.shift(-horizon).div(close) - 1.0
            one[f"ForwardDate{horizon}D"] = one["Date"].shift(-horizon)
        frames.append(one)
    return pd.concat(frames, ignore_index=True).sort_values(["Date", "Symbol"]).reset_index(drop=True)


def prepare_data(config: Dict[str, Any]) -> Dict[str, Any]:
    prepared_dir = Path(config["paths"]["prepared_dir"])
    prepared_dir.mkdir(parents=True, exist_ok=True)
    provider = config["data"]["provider"]
    source_files: dict[str, Any] = {}

    if provider == "csv":
        raw_path = Path(config["data"]["raw_prices_path"])
        prices, factors = load_csv_prices(raw_path)
        universes = configured_universes(config, prices)
        source_files["prices"] = {"path": str(raw_path), "sha256": sha256_file(raw_path)}
        membership_meta = None
    elif provider == "yfinance":
        membership = fetch_membership_snapshot(config, prepared_dir)
        membership_meta = {
            "url": config["data"]["membership"]["url"],
            "revision_id": config["data"]["membership"]["revision_id"],
            "as_of": config["data"]["membership"]["as_of"],
            "acquired_at_utc": membership.attrs["acquired_at_utc"],
            "sha256": membership.attrs["html_sha256"],
        }
        query_key = (
            f"{config['data']['source_start']}_{config['data']['source_end_exclusive']}"
            .replace("-", "")
        )
        price_cache_dir = prepared_dir / "price_chunks" / query_key
        prices = download_yfinance_prices(
            [*membership["Symbol"].tolist(), "SPY"],
            config["data"]["source_start"],
            config["data"]["source_end_exclusive"],
            cache_dir=price_cache_dir,
        )
        spy_dates = pd.DatetimeIndex(prices.loc[prices["Symbol"] == "SPY", "Date"].sort_values().unique())
        if spy_dates.empty:
            raise RuntimeError("SPY calendar could not be downloaded")
        prices = prices[prices["Symbol"] != "SPY"].merge(membership, on="Symbol", how="inner")
        universes = select_universes(config, membership, prices, spy_dates)
        selected = set(universes["train"] + universes["test"] + universes["reserve"])
        prices = prices[prices["Symbol"].isin(selected)].copy()
        factors = download_fama_french(
            config["data"]["source_start"],
            str(pd.Timestamp(config["data"]["source_end_exclusive"]) - pd.Timedelta(days=1)),
        )
        chunk_files = sorted(price_cache_dir.glob("chunk_*.csv.gz"))
        source_files["price_chunks"] = {
            "provider": "Yahoo Finance via yfinance",
            "url": YAHOO_FINANCE_URL,
            "query": {
                "start": config["data"]["source_start"],
                "end_exclusive": config["data"]["source_end_exclusive"],
                "adjustment": "OHLC multiplied by Adj Close / Close",
            },
            "directory": str(price_cache_dir),
            "files": [
                {
                    "name": path.name,
                    "cached_at_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                    "sha256": sha256_file(path),
                }
                for path in chunk_files
            ],
        }
        source_files["fama_french"] = {
            "provider": "Kenneth R. French Data Library via pandas-datareader",
            "dataset": FAMA_FRENCH_DATASET,
            "url": FAMA_FRENCH_URL,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    else:
        raise ValueError(f"Unsupported data provider: {provider}")

    prices["Date"] = pd.to_datetime(prices["Date"]).dt.tz_localize(None)
    factors["Date"] = pd.to_datetime(factors["Date"]).dt.tz_localize(None)
    panel = engineer_features(prices, config["data"]["target_horizons"])
    # Daily factor realizations are published after the trading day. Lag model
    # inputs by one factor observation; keep contemporaneous RF only for ex-post
    # performance measurement.
    factor_features = factors.copy()
    factor_features[config["data"]["macro_features"]] = factor_features[
        config["data"]["macro_features"]
    ].shift(1)
    panel = panel.merge(factor_features, on="Date", how="left", validate="many_to_one")
    if panel["RF"].isna().any():
        missing_dates = panel.loc[panel["RF"].isna(), "Date"].drop_duplicates().head(10).dt.strftime("%Y-%m-%d").tolist()
        raise RuntimeError(f"Risk-free rate is missing on stock dates, examples: {missing_dates}")
    macro_missing = panel[config["data"]["macro_features"]].isna().any(axis=1)
    allowed_initial_date = panel["Date"].min()
    unexpected_macro_missing = macro_missing & panel["Date"].ne(allowed_initial_date)
    if unexpected_macro_missing.any():
        missing_dates = panel.loc[unexpected_macro_missing, "Date"].drop_duplicates().head(10).dt.strftime("%Y-%m-%d").tolist()
        raise RuntimeError(f"Lagged Fama-French features are missing on stock dates, examples: {missing_dates}")

    selected_symbols = set(universes["train"] + universes["test"] + universes["reserve"])
    panel = panel[panel["Symbol"].isin(selected_symbols)].reset_index(drop=True)
    panel_path = prepared_dir / "panel.csv.gz"
    factors_path = prepared_dir / "fama_french_daily.csv"
    universe_path = prepared_dir / "universe_manifest.json"
    _deterministic_gzip_csv(panel, panel_path)
    factors.to_csv(factors_path, index=False)
    atomic_write_json(universe_path, universes)
    if "fama_french" in source_files:
        source_files["fama_french"].update(
            {"normalized_path": str(factors_path), "normalized_sha256": sha256_file(factors_path)}
        )

    if provider == "yfinance":
        train_set = set(universes["train"])
        test_set = set(universes["test"])
        reserve_set = set(universes["reserve"])
        expected_test = [_canonical_symbol(symbol) for symbol in config["data"]["universe"]["exact_test_symbols"]]
        sector_counts = pd.Series(
            [universes["sector_by_symbol"][symbol] for symbol in universes["train"]]
        ).value_counts()
        invalid = (
            len(train_set) != 55
            or len(test_set) != 10
            or len(reserve_set) != 5
            or universes["test"] != expected_test
            or bool(train_set & test_set)
            or bool(train_set & reserve_set)
            or bool(test_set & reserve_set)
            or len(sector_counts) != 11
            or not (sector_counts == 5).all()
            or len(universes["selection_status"]) != len(membership)
        )
        if invalid:
            raise RuntimeError(
                "Paper universe gate failed: expected exact 55/10/5 disjoint universes, "
                "eleven sectors with five train symbols each, and complete selection status"
            )

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(config["_meta"]["project_root"]),
        "data_configuration_sha256": data_configuration_hash(config),
        "provider": provider,
        "source_window": {
            "start": config["data"]["source_start"],
            "end_exclusive": config["data"]["source_end_exclusive"],
        },
        "factor_feature_lag_trading_days": 1,
        "membership": membership_meta,
        "source_files": source_files,
        "artifacts": {
            "panel": {"path": str(panel_path), "sha256": sha256_file(panel_path), "rows": len(panel)},
            "factors": {"path": str(factors_path), "sha256": sha256_file(factors_path), "rows": len(factors)},
            "universe": {"path": str(universe_path), "sha256": sha256_file(universe_path)},
        },
        "universe_counts": {key: len(universes[key]) for key in ("train", "test", "reserve", "n5", "n10", "n15")},
        "packages": package_versions(["pandas", "numpy", "yfinance", "pandas-datareader", "lxml"]),
    }
    manifest["data_hash"] = sha256_bytes(
        (manifest["artifacts"]["panel"]["sha256"] + manifest["artifacts"]["factors"]["sha256"] + manifest["artifacts"]["universe"]["sha256"]).encode("ascii")
    )
    atomic_write_json(prepared_dir / "data_manifest.json", manifest)
    return manifest


def load_prepared(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    prepared_dir = Path(config["paths"]["prepared_dir"])
    required = [
        prepared_dir / "panel.csv.gz",
        prepared_dir / "fama_french_daily.csv",
        prepared_dir / "universe_manifest.json",
        prepared_dir / "data_manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Prepared data artifacts are missing: {missing}. Run --stage prepare first.")
    panel = pd.read_csv(required[0], parse_dates=["Date"])
    for horizon in config["data"]["target_horizons"]:
        column = f"ForwardDate{horizon}D"
        if column in panel:
            panel[column] = pd.to_datetime(panel[column])
    factors = pd.read_csv(required[1], parse_dates=["Date"])
    universes = json.loads(required[2].read_text(encoding="utf-8"))
    manifest = json.loads(required[3].read_text(encoding="utf-8"))
    expected_configuration = data_configuration_hash(config)
    if manifest.get("data_configuration_sha256") != expected_configuration:
        raise RuntimeError("Prepared data was created with a different configuration; run --stage prepare again")
    current_hash = sha256_bytes(
        (sha256_file(required[0]) + sha256_file(required[1]) + sha256_file(required[2])).encode("ascii")
    )
    if current_hash != manifest["data_hash"]:
        raise RuntimeError("Prepared data hash differs from data_manifest.json")
    return panel, factors, universes, manifest
