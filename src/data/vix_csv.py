import pandas as pd
from pathlib import Path

def _best_date_parse(series: pd.Series) -> pd.Series:
    d_us = pd.to_datetime(series, errors="coerce", dayfirst=False)
    d_eu = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return d_eu if d_eu.notna().sum() > d_us.notna().sum() else d_us

def load_vix_csv(path: str) -> pd.DataFrame:
    """
    Lee VIX CSV con columnas típicas:
      DATE, OPEN, HIGH, LOW, CLOSE
    Devuelve DF indexado por fecha con columna 'vix' (usando CLOSE).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe: {path}")

    # autodetect separador
    head = p.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    sep = ";" if head.count(";") > head.count(",") else ","

    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip().lower() for c in df.columns]

    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(f"El VIX CSV debe tener DATE y CLOSE. Columnas: {df.columns.tolist()}")

    dates = _best_date_parse(df["date"])
    vix = pd.to_numeric(df["close"], errors="coerce")

    out = pd.DataFrame({"date": dates, "vix": vix}).dropna()
    out = out.set_index("date").sort_index()

    return out
