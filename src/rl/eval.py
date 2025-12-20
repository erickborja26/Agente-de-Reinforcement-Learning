import pandas as pd
from .env import TradingEnv
from ..utils.metrics import summarize_equity

def run_policy(model, data: pd.DataFrame, obs_cols, fee=0.0005) -> pd.Series:
    env = TradingEnv(data, obs_cols, fee=fee)
    obs, _ = env.reset()
    equities = [1.0]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, info = env.step(int(action))
        equities.append(info["equity"])
    return pd.Series(equities, index=data.index[:len(equities)], name="equity")

def evaluate(model, data: pd.DataFrame, obs_cols, fee=0.0005) -> dict:
    eq = run_policy(model, data, obs_cols, fee=fee)
    metrics = summarize_equity(eq)
    return {"equity": eq, "metrics": metrics}
