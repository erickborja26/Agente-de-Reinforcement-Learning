import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from src.config import settings
from src.data.build_dataset import build_dataset
from src.hmm.hmm_regime import fit_hmm_regimes
from src.utils.io import ensure_dir, save_csv


def plot_hmm_probabilities(df_hmm: pd.DataFrame, out_png: Path, n_states: int):
    pcols = [f"hmm_p{k}" for k in range(n_states)]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.stackplot(df_hmm.index, *[df_hmm[c].values for c in pcols], labels=pcols)
    ax.set_title("HMM - Probabilidades de régimen en el tiempo")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Probabilidad")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_price_with_states(df_hmm: pd.DataFrame, out_png: Path):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_hmm.index, df_hmm["close"].values, label="EPU Close")

    # Pinta puntos por estado (simple y claro)
    for st in sorted(df_hmm["hmm_state"].unique()):
        mask = df_hmm["hmm_state"] == st
        ax.scatter(df_hmm.index[mask], df_hmm.loc[mask, "close"], s=6, label=f"state={st}")

    ax.set_title("EPU (close) con estados HMM")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio")
    ax.legend(loc="upper left", ncols=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    # 1) Reconstruir dataset (4 fuentes) usando tu pipeline actual
    processed_path = Path(settings.processed_dir) / "master_df.csv"

    if processed_path.exists():
        print(f"Cargando dataset ya procesado: {processed_path}")
        df = pd.read_csv(processed_path, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()
    else:
        # 2) Si no existe, construirlo una vez y guardarlo
        print("No existe master_dataset.csv -> construyendo dataset...")
        df = build_dataset(
            ticker=settings.ticker,
            start=settings.start,
            end=settings.end,
            vix_csv_path=settings.vix_csv_path,
            wb_country=settings.wb_country,
            wb_indicators=settings.wb_indicators,
            av_base_url=settings.alphavantage_base_url,
            av_api_key=settings.alphavantage_api_key,
            av_topics=settings.alphavantage_topics,
            av_sort=settings.alphavantage_sort,
            av_limit=settings.alphavantage_limit,
            av_sleep_sec=settings.alphavantage_sleep_sec,
            cache_dir=settings.cache_dir,
            processed_dir=settings.processed_dir,
            save_name="master_df.csv"
        )
        
         # Guardar para futuras corridas
        ensure_dir(settings.processed_dir)
        out = df.copy()
        out.index.name = "Date"
        out.reset_index().to_csv(processed_path, index=False)
        print(f"Guardado dataset procesado: {processed_path}")

    # 2) Features HMM SOLO con precio + volatilidad (como pide el enunciado)
    hmm_cols = ["ret", "vol_20", "mom_5", "vix"]
    df = df.dropna(subset=["close"] + hmm_cols).copy()

    # 3) Fit HMM en todo el dataset (solo para visualización/reporte)
    hmm_res = fit_hmm_regimes(
        df,
        feature_cols=hmm_cols,
        n_states=settings.hmm_states,
        seed=settings.hmm_seed
    )
    df_hmm = hmm_res.df

    # 4) Exportar CSV con probabilidades
    out_hmm_dir = Path(settings.artifacts_dir) / "hmm"
    out_plot_dir = Path(settings.artifacts_dir) / "plots"
    ensure_dir(str(out_hmm_dir))
    ensure_dir(str(out_plot_dir))

    pcols = [f"hmm_p{k}" for k in range(settings.hmm_states)]
    export_cols = ["close", "hmm_state"] + pcols

    df_hmm_out = df_hmm[export_cols].copy()
    df_hmm_out.index.name = "Date"
    df_hmm_out.reset_index(inplace=True)

    save_csv(df_hmm_out, str(out_hmm_dir / "hmm_probabilities_timeseries.csv"), index=False)

    # 5) Exportar también transición y emisión (por si lo quieres pegar al informe)
    save_csv(hmm_res.transition, str(out_hmm_dir / "transition.csv"), index=False)
    save_csv(hmm_res.emission_means, str(out_hmm_dir / "emission_means.csv"), index=False)

    # 6) Resumen de estados
    state_counts = df_hmm["hmm_state"].value_counts(normalize=True).sort_index()
    state_counts_df = state_counts.rename("proportion").reset_index().rename(columns={"index": "hmm_state"})
    save_csv(state_counts_df, str(out_hmm_dir / "state_counts.csv"), index=False)

    # 7) Gráficos
    plot_hmm_probabilities(df_hmm, out_plot_dir / "hmm_probabilities.png", settings.hmm_states)
    plot_price_with_states(df_hmm, out_plot_dir / "price_with_hmm_states.png")

    print("\n[OK] Exportados:")
    print(f"- {out_hmm_dir / 'hmm_probabilities_timeseries.csv'}")
    print(f"- {out_plot_dir / 'hmm_probabilities.png'}")
    print(f"- {out_plot_dir / 'price_with_hmm_states.png'}")
    print(f"- {out_hmm_dir / 'transition.csv'}")
    print(f"- {out_hmm_dir / 'emission_means.csv'}")
    print(f"- {out_hmm_dir / 'state_counts.csv'}")


if __name__ == "__main__":
    main()
