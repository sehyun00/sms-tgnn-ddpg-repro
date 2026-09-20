from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


def validate_weights(weights: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Portfolio weights must be a non-empty vector")
    if not np.isfinite(values).all():
        raise RuntimeError("Portfolio weights contain NaN or infinity")
    if (values < -tolerance).any():
        raise RuntimeError("Long-only portfolio produced a negative weight")
    values = np.maximum(values, 0.0)
    total = values.sum()
    if total <= 0.0 or not np.isclose(total, 1.0, atol=tolerance):
        raise RuntimeError(f"Portfolio weights do not sum to one: {total}")
    return values / total


def drift_weights(weights: np.ndarray, asset_returns: np.ndarray) -> np.ndarray:
    gross_assets = np.asarray(weights) * (1.0 + np.asarray(asset_returns))
    total = gross_assets.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("Portfolio value became non-positive while drifting weights")
    return gross_assets / total


def portfolio_turnover(target: np.ndarray, pretrade: np.ndarray | None) -> float:
    target = validate_weights(target)
    if pretrade is None:
        return 1.0
    pretrade = validate_weights(pretrade)
    return float(0.5 * np.abs(target - pretrade).sum())


@dataclass
class BacktestOutput:
    daily: pd.DataFrame
    decisions: pd.DataFrame


def run_weight_backtest(
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
    cost_bps: float,
    initial_capital: float = 1_000_000.0,
) -> BacktestOutput:
    """Apply one gross weight path to one explicit transaction-cost scenario."""
    if close.empty or target_weights.empty:
        raise ValueError("Close prices and target weights must be non-empty")
    close = close.sort_index().astype(float)
    target_weights = target_weights.sort_index().astype(float)
    if list(close.columns) != list(target_weights.columns):
        raise ValueError("Close and target-weight columns must match in the same order")
    if not target_weights.index.isin(close.index).all():
        missing = target_weights.index[~target_weights.index.isin(close.index)].tolist()
        raise ValueError(f"Target-weight dates are absent from close prices: {missing[:5]}")
    if float(cost_bps) < 0:
        raise ValueError("Transaction cost cannot be negative")

    start_position = close.index.get_loc(target_weights.index[0])
    current_weights: np.ndarray | None = None
    nav = float(initial_capital)
    daily_rows: list[dict[str, float | str]] = []
    decision_rows: list[dict[str, float | str]] = []

    for position in range(start_position, len(close.index) - 1):
        date = close.index[position]
        next_date = close.index[position + 1]
        pending_cost = 0.0
        turnover = 0.0
        if date in target_weights.index:
            target = validate_weights(target_weights.loc[date].to_numpy())
            turnover = portfolio_turnover(target, current_weights)
            pending_cost = turnover * float(cost_bps) / 10_000.0
            current_weights = target
            decision_rows.append(
                {
                    "decision_date": date,
                    "turnover": turnover,
                    "cost_bps": float(cost_bps),
                    "cost_fraction": pending_cost,
                }
            )
        if current_weights is None:
            continue

        asset_returns = close.iloc[position + 1].to_numpy() / close.iloc[position].to_numpy() - 1.0
        if not np.isfinite(asset_returns).all() or (asset_returns <= -1.0).any():
            raise RuntimeError(f"Invalid asset return between {date.date()} and {next_date.date()}")
        gross_return = float(np.dot(current_weights, asset_returns))
        net_return = float((1.0 - pending_cost) * (1.0 + gross_return) - 1.0)
        if net_return <= -1.0 or not np.isfinite(net_return):
            raise RuntimeError(f"Invalid portfolio return at {next_date.date()}: {net_return}")
        nav *= 1.0 + net_return
        daily_rows.append(
            {
                "Date": next_date,
                "gross_return": gross_return,
                "cost_fraction": pending_cost,
                "net_return": net_return,
                "nav": nav,
                "turnover": turnover,
            }
        )
        current_weights = drift_weights(current_weights, asset_returns)

    daily = pd.DataFrame(daily_rows).set_index("Date")
    decisions = pd.DataFrame(decision_rows)
    if daily.empty:
        raise RuntimeError("Backtest produced no daily returns")
    return BacktestOutput(daily=daily, decisions=decisions)


def compute_metrics(
    daily: pd.DataFrame, risk_free: pd.Series, decisions: pd.DataFrame | None = None
) -> dict[str, float]:
    returns = daily["net_return"].astype(float)
    aligned_rf = risk_free.reindex(returns.index)
    if aligned_rf.isna().any():
        dates = aligned_rf[aligned_rf.isna()].index[:5].strftime("%Y-%m-%d").tolist()
        raise RuntimeError(f"Risk-free rate missing on backtest dates: {dates}")
    if len(returns) < 2:
        raise RuntimeError("At least two daily returns are required for metrics")
    elapsed_start = (
        pd.Timestamp(decisions.iloc[0]["decision_date"])
        if decisions is not None and not decisions.empty
        else returns.index[0]
    )
    elapsed_days = (returns.index[-1] - elapsed_start).days
    if elapsed_days <= 0:
        raise RuntimeError("Backtest dates do not span positive elapsed time")
    wealth = (1.0 + returns).cumprod()
    total_return = float(wealth.iloc[-1] - 1.0)
    years = elapsed_days / 365.2425
    cagr = float(wealth.iloc[-1] ** (1.0 / years) - 1.0)
    excess = returns - aligned_rf.astype(float)
    volatility = float(returns.std(ddof=1) * np.sqrt(252.0))
    excess_std = float(excess.std(ddof=1))
    sharpe = float(excess.mean() / excess_std * np.sqrt(252.0)) if excess_std > 0 else float("nan")
    drawdown = wealth / wealth.cummax() - 1.0
    turnover_values = (
        decisions["turnover"].astype(float)
        if decisions is not None and not decisions.empty
        else daily["turnover"].astype(float)
    )
    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "mdd": float(drawdown.min()),
        "annualized_volatility": volatility,
        "mean_turnover": float(turnover_values.mean()),
        "total_turnover": float(turnover_values.sum()),
        "observations": int(len(returns)),
    }


def shrinkage_covariance(trailing_returns: pd.DataFrame) -> np.ndarray:
    clean = trailing_returns.dropna(how="any")
    if len(clean) < 2:
        raise RuntimeError("Insufficient complete observations for covariance estimation")
    covariance = LedoitWolf().fit(clean.to_numpy()).covariance_
    if not np.isfinite(covariance).all():
        raise RuntimeError("Ledoit-Wolf covariance contains non-finite values")
    return covariance


def minimum_variance_weights(covariance: np.ndarray) -> np.ndarray:
    n_assets = covariance.shape[0]
    objective: Callable[[np.ndarray], float] = lambda weights: float(weights @ covariance @ weights)
    result = minimize(
        objective,
        np.full(n_assets, 1.0 / n_assets),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Minimum-variance optimization failed: {result.message}")
    return validate_weights(result.x)


def equal_risk_contribution_weights(covariance: np.ndarray) -> np.ndarray:
    n_assets = covariance.shape[0]
    volatility = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    initial = (1.0 / volatility) / (1.0 / volatility).sum()

    def objective(weights: np.ndarray) -> float:
        portfolio_variance = float(weights @ covariance @ weights)
        if portfolio_variance <= 0:
            return 1e9
        marginal = covariance @ weights
        contributions = weights * marginal / portfolio_variance
        return float(np.square(contributions - 1.0 / n_assets).sum())

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(1e-8, 1.0)] * n_assets,
        constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Equal-risk-contribution optimization failed: {result.message}")
    return validate_weights(result.x)


def baseline_weights(
    close: pd.DataFrame,
    decision_dates: Sequence[pd.Timestamp],
    method: str,
    lookback: int,
) -> pd.DataFrame:
    rows: list[np.ndarray] = []
    valid_dates: list[pd.Timestamp] = []
    returns = close.pct_change(fill_method=None)
    for date in decision_dates:
        position = close.index.get_loc(date)
        if method == "equal_weight":
            weights = np.full(close.shape[1], 1.0 / close.shape[1])
        else:
            trailing = returns.iloc[max(1, position - lookback + 1) : position + 1]
            if len(trailing.dropna(how="any")) < lookback:
                raise RuntimeError(
                    f"{method} requires {lookback} complete observations before {date.date()}"
                )
            covariance = shrinkage_covariance(trailing)
            if method == "min_variance":
                weights = minimum_variance_weights(covariance)
            elif method == "risk_parity":
                weights = equal_risk_contribution_weights(covariance)
            else:
                raise ValueError(f"Unknown baseline: {method}")
        rows.append(weights)
        valid_dates.append(pd.Timestamp(date))
    return pd.DataFrame(rows, index=pd.DatetimeIndex(valid_dates), columns=close.columns)


def interval_net_returns(
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
    cost_bps: float,
) -> pd.Series:
    """Return one net cumulative outcome per rebalance decision for alpha labels."""
    decisions = list(target_weights.index)
    values: dict[pd.Timestamp, float] = {}
    pretrade: np.ndarray | None = None
    for index, date in enumerate(decisions[:-1]):
        next_date = decisions[index + 1]
        target = validate_weights(target_weights.loc[date].to_numpy())
        turnover = portfolio_turnover(target, pretrade)
        cost = turnover * float(cost_bps) / 10_000.0
        asset_returns = close.loc[next_date].to_numpy() / close.loc[date].to_numpy() - 1.0
        gross = float(np.dot(target, asset_returns))
        net = float((1.0 - cost) * (1.0 + gross) - 1.0)
        values[pd.Timestamp(date)] = net
        pretrade = drift_weights(target, asset_returns)
    return pd.Series(values, name="interval_net_return", dtype=float)
