from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from .env import TradingEnv

def train_dqn(train_df, obs_cols, timesteps=50_000, seed=7, fee=0.0005) -> DQN:
    env = DummyVecEnv([lambda: TradingEnv(train_df, obs_cols, fee=fee)])
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=5_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        verbose=0,
        seed=seed
    )
    model.learn(total_timesteps=timesteps)
    return model
