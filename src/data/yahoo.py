import pandas as pd
import yfinance as yf

def load_yahoo_close(ticker: str, start: str, end: str) -> pd.DataFrame:
    # group_by="column" suele evitar MultiIndex, pero igual lo manejamos por si aparece
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        group_by="column",
        progress=False
    )

    if df is None or df.empty:
        raise RuntimeError(f"No se descargó data de Yahoo para ticker={ticker}")

    df.index = pd.to_datetime(df.index)
    tk = str(ticker).upper()

    # --- Caso 1: MultiIndex (dos niveles) ---
    if isinstance(df.columns, pd.MultiIndex):
        # Normalizamos niveles: nivel0=campo (close/open/etc), nivel1=ticker
        lvl0 = df.columns.get_level_values(0).astype(str).str.lower()
        lvl1 = df.columns.get_level_values(1).astype(str).str.upper()

        # Intentar encontrar exactamente ('close', ticker)
        mask = (lvl0 == "close") & (lvl1 == tk)
        if mask.any():
            close_series = df.loc[:, mask].iloc[:, 0]
        else:
            # Fallback: tomar cualquier "close" disponible
            mask2 = (lvl0 == "close")
            if not mask2.any():
                raise RuntimeError(f"No existe 'close' en Yahoo. Columnas: {df.columns.tolist()}")
            close_series = df.loc[:, mask2].iloc[:, 0]

        out = close_series.to_frame(name="close")
        return out.sort_index()

    # --- Caso 2: columnas normales (un nivel) ---
    cols = {str(c).lower(): c for c in df.columns}

    # A veces puede venir como "Adj Close" si auto_adjust=False
    if "close" in cols:
        out = df[[cols["close"]]].rename(columns={cols["close"]: "close"})
        return out.sort_index()

    if "adj close" in cols:
        out = df[[cols["adj close"]]].rename(columns={cols["adj close"]: "close"})
        return out.sort_index()

    raise RuntimeError(f"No existe columna Close en Yahoo. Columnas: {df.columns.tolist()}")
