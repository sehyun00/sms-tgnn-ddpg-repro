import numpy as np
import pandas as pd
from typing import Dict, List, Union


def compute_metrics(
    portfolio_values: Union[List[float], np.ndarray],
    initial_capital: float = 1_000_000,
    risk_free_rate: float = 0.02,  # 연간 무위험수익률 (기본 2%)
) -> Dict[str, float]:
    """
    Computes financial metrics (Return, CAGR, MDD, Sharpe) from portfolio history.

    Args:
        portfolio_values: 포트폴리오 가치 시계열
        initial_capital: 초기 자본
        risk_free_rate: 연간 무위험수익률 (기본 2%)
    """
    values = np.array(portfolio_values)
    if len(values) == 0:
        return {"Total_Return": 0.0, "CAGR": 0.0, "MDD": 0.0, "Sharpe": 0.0}

    # Returns
    total_ret = (values[-1] / initial_capital) - 1

    # CAGR (Annualized) - Assuming Daily Steps
    days = len(values)  # Trading Days
    years = days / 252.0
    if years > 0:
        cagr = (values[-1] / initial_capital) ** (1 / years) - 1
    else:
        cagr = 0

    # MDD
    running_max = np.maximum.accumulate(values)
    drawdown = (values - running_max) / running_max
    mdd = abs(np.min(drawdown))

    # Sharpe Ratio (무위험수익률 반영)
    pct_change = pd.Series(values).pct_change().dropna()
    if len(pct_change) > 0:
        vol = pct_change.std() * np.sqrt(252)
        excess_return = cagr - risk_free_rate  # 초과수익률 = CAGR - RF
        sharpe = excess_return / (vol + 1e-8)
    else:
        sharpe = 0

    return {
        "Total_Return": total_ret * 100,
        "CAGR": cagr * 100,
        "MDD": mdd * 100,
        "Sharpe": sharpe,
    }
