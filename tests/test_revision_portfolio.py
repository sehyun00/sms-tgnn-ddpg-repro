from __future__ import annotations

import numpy as np
import pandas as pd

from src.revision.backtest import backtest_specification
from src.revision.config import load_revision_config
from src.revision.portfolio import (
    compute_metrics,
    drift_weights,
    portfolio_turnover,
    run_weight_backtest,
)


def test_secondary_graph_and_variable_universe_runs_only_use_primary_cost():
    config = load_revision_config("config/paper_revision.yaml")
    secondary = backtest_specification(
        config, "secondary", "equal_weight", 0, "quarterly", "none", "n10"
    )
    graph = backtest_specification(
        config, "primary", "tgnn", 42, "quarterly", "sector", "n10"
    )
    variable_n = backtest_specification(
        config, "primary", "ddpg", 42, "quarterly", "none", "n5"
    )
    core = backtest_specification(
        config, "primary", "hybrid", 42, "quarterly", "combined", "n10"
    )
    assert secondary["cost_bps"] == graph["cost_bps"] == variable_n["cost_bps"] == [10.0]
    assert core["cost_bps"] == [0.0, 5.0, 10.0, 20.0]


def test_turnover_uses_initial_one_and_drifted_pretrade_weights():
    target = np.array([0.5, 0.5])
    assert portfolio_turnover(target, None) == 1.0
    drifted = drift_weights(target, np.array([0.10, 0.0]))
    expected = 0.5 * np.abs(target - drifted).sum()
    assert np.isclose(portfolio_turnover(target, drifted), expected)


def test_cost_sweep_reuses_identical_gross_path():
    dates = pd.bdate_range("2021-01-04", periods=5)
    close = pd.DataFrame(
        {"AAA": [100, 101, 102, 103, 104], "BBB": [100, 99, 98, 99, 100]}, index=dates
    )
    weights = pd.DataFrame(
        [[0.5, 0.5], [0.8, 0.2]], index=[dates[0], dates[2]], columns=close.columns
    )
    zero = run_weight_backtest(close, weights, 0)
    ten = run_weight_backtest(close, weights, 10)
    assert np.allclose(zero.daily["gross_return"], ten.daily["gross_return"])
    assert ten.daily["net_return"].sum() < zero.daily["net_return"].sum()
    assert np.isclose(ten.decisions.iloc[0]["turnover"], 1.0)


def test_metrics_align_daily_risk_free_and_actual_elapsed_time():
    dates = pd.bdate_range("2021-01-04", periods=30)
    returns = np.linspace(-0.002, 0.003, len(dates))
    daily = pd.DataFrame(
        {
            "net_return": returns,
            "turnover": np.zeros(len(dates)),
            "nav": 1_000_000 * np.cumprod(1 + returns),
        },
        index=dates,
    )
    start_date = dates[0] - pd.offsets.BDay(1)
    decisions = pd.DataFrame({"decision_date": [start_date], "turnover": [1.0]})
    metrics = compute_metrics(daily, pd.Series(0.00001, index=dates), decisions)
    expected_years = (dates[-1] - start_date).days / 365.2425
    expected_cagr = float(np.prod(1.0 + returns) ** (1.0 / expected_years) - 1.0)
    assert np.isfinite(metrics["sharpe"])
    assert np.isclose(metrics["cagr"], expected_cagr)
    assert metrics["observations"] == 30
    assert metrics["mdd"] <= 0.0
