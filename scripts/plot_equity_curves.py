import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from src.config import settings
from src.utils.io import ensure_dir


def load_equity_series(path: Path, name: str) -> pd.Series:
    """
    Soporta dos formatos típicos:
    1) CSV con una columna equity y el índice como fecha.
    2) CSV con columnas [date, equity] o [Date, equity].
    """
    if not path.exists():
        raise FileNotFoundError(f"No existe archivo: {path}")

    df = pd.read_csv(path)

    # Caso: tiene columna equity y una columna date
    cols_lower = {c.lower(): c for c in df.columns}
    if "equity" in cols_lower:
        eq_col = cols_lower["equity"]

        # buscar fecha
        date_col = None
        for key in ["date", "datetime", "timestamp", "unnamed: 0", "index"]:
            if key in cols_lower:
                date_col = cols_lower[key]
                break
        if date_col is None:
            # si no hay columna fecha, usar la primera como fecha
            date_col = df.columns[0]

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        s = df[eq_col].astype(float)
        s.name = name
        return s

    # Caso: CSV guardado como "serie" (1 columna) con índice
    if df.shape[1] == 1:
        s = df.iloc[:, 0].astype(float)
        s.index = pd.to_datetime(df.index, errors="coerce")
        s = s.dropna()
        s.name = name
        return s

    raise ValueError(f"No pude interpretar formato de equity en {path}. Columnas: {df.columns.tolist()}")


def load_master_dataset(path: Path) -> pd.DataFrame:
    """
    Carga el master dataset aunque la fecha se llame date/Date/Unnamed: 0.
    """
    if not path.exists():
        raise FileNotFoundError(f"No existe dataset procesado: {path}")

    raw = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in raw.columns}

    date_col = None
    for key in ["date", "datetime", "timestamp", "unnamed: 0", "index"]:
        if key in cols_lower:
            date_col = cols_lower[key]
            break
    if date_col is None:
        date_col = raw.columns[0]

    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col]).set_index(date_col).sort_index()
    raw.index.name = "date"
    return raw


def main():
    # Rutas esperadas
    eq_hmm_path = Path(settings.artifacts_dir) / "equity_hmm.csv"
    eq_no_path = Path(settings.artifacts_dir) / "equity_nohmm.csv"
    master_path = Path(settings.processed_dir) / "master_df.csv"

    # Cargar equities de agentes
    eq_hmm = load_equity_series(eq_hmm_path, "DQN + HMM")
    eq_no = load_equity_series(eq_no_path, "DQN sin HMM")

    # Baseline Buy&Hold: se calcula desde master_dataset.csv
    master = load_master_dataset(master_path)

    # Alinear el baseline al rango común de fechas que tengan los equities
    start = max(eq_hmm.index.min(), eq_no.index.min())
    end = min(eq_hmm.index.max(), eq_no.index.max())
    master_slice = master.loc[start:end].copy()

    if "close" not in master_slice.columns:
        raise ValueError("El master_dataset.csv debe tener columna 'close' para calcular Buy&Hold.")

    bh = (master_slice["close"] / master_slice["close"].iloc[0]).rename("Buy & Hold")

    # Alinear series por fechas (inner join)
    df_plot = pd.concat([bh, eq_hmm, eq_no], axis=1).dropna()

    # Plot
    out_dir = Path(settings.artifacts_dir) / "plots"
    ensure_dir(str(out_dir))
    out_png = out_dir / "equity_curves.png"

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_plot.index, df_plot["Buy & Hold"], label="Buy & Hold")
    ax.plot(df_plot.index, df_plot["DQN + HMM"], label="DQN + HMM")
    ax.plot(df_plot.index, df_plot["DQN sin HMM"], label="DQN sin HMM")

    ax.set_title("Equity Curves - Buy&Hold vs DQN+HMM vs DQN sin HMM")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Equity (normalizada)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print("\n[OK] Guardado gráfico:")
    print(out_png)


if __name__ == "__main__":
    main()
