import sys
from pathlib import Path

# Asegura que el root del proyecto esté en sys.path (para imports de src)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import settings
from src.data.build_dataset import build_dataset
from src.hmm.hmm_regime import fit_hmm_regimes
from src.features.scaling import scale_columns
from src.rl.train import train_dqn
from src.rl.eval import evaluate
from src.utils.io import ensure_dir, save_joblib, save_csv

def main():
    # 1) Construir dataset (4 fuentes)
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

    # Features base (sin HMM) – usa macro WB + VIX + sentiment + features de precio
    base_cols = ["ret", "vol_20", "mom_5"] + settings.wb_indicators + ["vix", "sentiment"]
    df = df.dropna(subset=base_cols).copy()

    # 2) HMM
    hmm_res = fit_hmm_regimes(df, feature_cols=base_cols, n_states=settings.hmm_states, seed=settings.hmm_seed)
    df_hmm = hmm_res.df

    # Imprimir lo pedido: transición y emisión (medias)
    print("\n=== HMM Transition Matrix ===")
    print(hmm_res.transition)

    print("\n=== HMM Emission Means (scaled-space means) ===")
    print(hmm_res.emission_means)

    print("\n=== HMM State Labels (por retorno medio) ===")
    print(hmm_res.state_labels)

    # 3) Split train/test
    split = int(len(df_hmm) * settings.rl_train_ratio)
    train_df = df_hmm.iloc[:split].copy()
    test_df = df_hmm.iloc[split:].copy()

    # 4) Escalado para RL (solo columnas base; hmm_p* se dejan tal cual)
    scaled = scale_columns(train_df, test_df, cols=base_cols)
    train_s = scaled.train_df
    test_s = scaled.test_df

    # Baseline Buy & Hold en test
    test_prices = test_s["close"].copy()
    bh_eq = (test_prices / test_prices.iloc[0]).rename("equity")
    bh_metrics = {
        "Cumulative Return": float(bh_eq.iloc[-1] - 1.0),
        "Sharpe": float((bh_eq.pct_change().mean() / bh_eq.pct_change().std()) * (252**0.5)),
        "Max Drawdown": float(((bh_eq / bh_eq.cummax()) - 1.0).min()),
    }
    print("\n=== Baseline Buy&Hold (TEST) ===")
    print(bh_metrics)

    # Observaciones RL
    obs_nohmm = base_cols
    obs_hmm = base_cols + [f"hmm_p{k}" for k in range(settings.hmm_states)]

    # 5) Entrenar DQN con/sin HMM
    print("\nEntrenando DQN + HMM ...")
    model_hmm = train_dqn(
        train_s, obs_hmm,
        timesteps=settings.rl_timesteps,
        seed=settings.rl_seed,
        fee=settings.rl_fee
    )

    print("Entrenando DQN sin HMM ...")
    model_no = train_dqn(
        train_s, obs_nohmm,
        timesteps=settings.rl_timesteps,
        seed=settings.rl_seed,
        fee=settings.rl_fee
    )

    # 6) Evaluación (Sharpe, CumReturn, MaxDD) con vs sin HMM
    eval_hmm = evaluate(model_hmm, test_s, obs_hmm, fee=settings.rl_fee)
    eval_no = evaluate(model_no, test_s, obs_nohmm, fee=settings.rl_fee)

    results = pd.DataFrame({
        "DQN + HMM": eval_hmm["metrics"],
        "DQN sin HMM": eval_no["metrics"]
    }).T

    print("\n=== RESULTADOS (TEST) ===")
    print(results)

    # 7) Guardar artifacts
    ensure_dir(settings.artifacts_dir)
    ensure_dir(str(Path(settings.artifacts_dir) / "scalers"))
    ensure_dir(str(Path(settings.artifacts_dir) / "hmm"))
    ensure_dir(str(Path(settings.artifacts_dir) / "rl"))

    # métricas y equity
    save_csv(results, str(Path(settings.artifacts_dir) / "metrics.csv"))
    eval_hmm["equity"].to_csv(str(Path(settings.artifacts_dir) / "equity_hmm.csv"))
    eval_no["equity"].to_csv(str(Path(settings.artifacts_dir) / "equity_nohmm.csv"))

    # HMM
    save_csv(hmm_res.transition, str(Path(settings.artifacts_dir) / "hmm" / "transition.csv"), index=False)
    save_csv(hmm_res.emission_means, str(Path(settings.artifacts_dir) / "hmm" / "emission_means.csv"), index=False)
    save_joblib(hmm_res.model, str(Path(settings.artifacts_dir) / "hmm" / "hmm_model.joblib"))
    save_joblib(hmm_res.scaler, str(Path(settings.artifacts_dir) / "hmm" / "hmm_scaler.joblib"))

    # scaler RL
    save_joblib(scaled.scaler, str(Path(settings.artifacts_dir) / "scalers" / "rl_scaler.joblib"))

    # RL models (SB3)
    model_hmm.save(str(Path(settings.artifacts_dir) / "rl" / "dqn_hmm"))
    model_no.save(str(Path(settings.artifacts_dir) / "rl" / "dqn_nohmm"))

    print(f"\nArtifacts guardados en: {Path(settings.artifacts_dir).resolve()}")

if __name__ == "__main__":
    main()
