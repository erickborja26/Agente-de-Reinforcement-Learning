import pandas as pd
from pathlib import Path

def load_vix_csv(path: str) -> pd.DataFrame:
    """
    Lee VIX CSV con columnas típicas:
      DATE, OPEN, HIGH, LOW, CLOSE
    y devuelve DF indexado por fecha con columna 'vol' (= CLOSE).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(f"Esperaba columnas DATE y CLOSE. Columnas: {df.columns.tolist()}")

    # Parse robusto de fecha (mm/dd vs dd/mm)
    d_us = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)
    d_eu = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    dates = d_eu if d_eu.notna().sum() > d_us.notna().sum() else d_us

    out = pd.DataFrame({
        "date": dates,
        "vol": pd.to_numeric(df["close"], errors="coerce")
    }).dropna()

    out = out.set_index("date").sort_index()

    # Si quieres vol en 0-1 en vez de 0-100, descomenta:
    # out["vol"] = out["vol"] / 100.0

    return out
