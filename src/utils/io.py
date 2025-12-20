from pathlib import Path
import joblib
import pandas as pd

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def save_joblib(obj, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    joblib.dump(obj, path)

def load_joblib(path: str):
    return joblib.load(path)

def save_csv(df: pd.DataFrame, path: str, index=True) -> None:
    ensure_dir(str(Path(path).parent))
    df.to_csv(path, index=index)
