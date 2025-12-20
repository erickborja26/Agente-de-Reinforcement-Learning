import numpy as np
import pandas as pd

def sharpe_ratio(equity: pd.Series, periods: int = 252) -> float:
    r = equity.pct_change().dropna()
    if r.std() == 0 or np.isnan(r.std()):
        return float("nan")
    return float(np.sqrt(periods) * (r.mean() / r.std()))

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())

def cumulative_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] - 1.0)

def summarize_equity(equity: pd.Series) -> dict:
    return {
        "Cumulative Return": cumulative_return(equity),
        "Sharpe": sharpe_ratio(equity),
        "Max Drawdown": max_drawdown(equity),
    }
