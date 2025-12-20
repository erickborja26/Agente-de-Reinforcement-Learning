import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.rl.env import TradingEnv

def action_counts(model, data: pd.DataFrame, obs_cols, fee=0.0005):
    env = TradingEnv(data, obs_cols, fee=fee)
    obs, _ = env.reset()
    done = False
    actions = []
    equities = [1.0]

    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, info = env.step(int(a))
        actions.append(int(a))
        equities.append(info["equity"])

    actions = np.array(actions)
    counts = {
        "Hold(0)": int((actions == 0).sum()),
        "Buy(1)": int((actions == 1).sum()),
        "Sell(2)": int((actions == 2).sum()),
        "Total": int(len(actions))
    }
    eq = pd.Series(equities, index=data.index[:len(equities)], name="equity")
    return counts, eq
