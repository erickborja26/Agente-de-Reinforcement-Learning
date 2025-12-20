import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    """
    Acciones:
      0 = Hold
      1 = Buy  (posición long +1)
      2 = Sell (posición short -1)
    Reward = pos * retorno - costo_transacción
    """
    metadata = {"render_modes": []}

    def __init__(self, data, obs_cols, price_col="close", fee=0.0005):
        super().__init__()
        self.data = data
        self.obs_cols = obs_cols
        self.price_col = price_col
        self.fee = fee

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(len(obs_cols),), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.position = 0
        self.equity = 1.0
        return self._obs(), {}

    def _obs(self):
        return self.data.iloc[self.t][self.obs_cols].values.astype(np.float32)

    def step(self, action):
        prev_price = float(self.data.iloc[self.t][self.price_col])
        self.t += 1
        terminated = self.t >= (len(self.data) - 1)
        curr_price = float(self.data.iloc[self.t][self.price_col])

        ret = (curr_price / prev_price) - 1.0

        target_pos = {0: self.position, 1: 1, 2: -1}[int(action)]
        turnover = abs(target_pos - self.position)
        cost = turnover * self.fee
        self.position = target_pos

        reward = (self.position * ret) - cost
        self.equity *= (1.0 + reward)

        info = {"equity": self.equity, "position": self.position}
        return self._obs(), float(reward), terminated, False, info
