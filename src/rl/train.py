import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from .env import TradingEnv

def train_dqn(train_df, obs_cols, timesteps=50_000):
    env = DummyVecEnv([lambda: TradingEnv(train_df, obs_cols)])
    model = DQN(
        "MlpPolicy", env,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=5_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        verbose=0,
    )
    model.learn(total_timesteps=timesteps)
    return model

def run_policy(model, data, obs_cols):
    env = TradingEnv(data, obs_cols)
    obs, _ = env.reset()
    equities = [1.0]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, info = env.step(int(action))
        equities.append(info["equity"])
    eq = pd.Series(equities, index=data.index[:len(equities)])
    return eq

def sharpe(eq: pd.Series, periods=252):
    r = eq.pct_change().dropna()
    return np.sqrt(periods) * (r.mean() / r.std()) if r.std() != 0 else np.nan

def max_drawdown(eq: pd.Series):
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())

def summarize(eq: pd.Series):
    return {
        "Cumulative Return": float(eq.iloc[-1] - 1.0),
        "Sharpe": float(sharpe(eq)),
        "Max Drawdown": float(max_drawdown(eq)),
    }
