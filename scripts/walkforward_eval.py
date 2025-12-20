import sys
from pathlib import Path

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
from src.utils.io import ensure_dir, save_csv
from src.utils.metrics import summarize_equity

from scripts.walkforward_debug_actions import action_counts


def make_windows(df: pd.DataFrame, splits: list[tuple[str, str, str, str]]):
    """
    splits = [(train_start, train_end, test_start, test_end), ...]
    devuelve lista de dicts con train_df/test_df recortados por fecha
    """
    windows = []
    for tr0, tr1, te0, te1 in splits:
        train_df = df.loc[tr0:tr1].copy()
        test_df = df.loc[te0:te1].copy()

        # Evitar ventanas vacías
        if len(train_df) < 300 or len(test_df) < 60:
            continue

        windows.append({
            "train_start": tr0, "train_end": tr1,
            "test_start": te0, "test_end": te1,
            "train_df": train_df, "test_df": test_df
        })
    return windows


def main():
    # 1) Dataset completo (4 fuentes)
    processed_path = Path(settings.processed_dir) / "master_df.csv"

    if processed_path.exists():
        df = pd.read_csv(processed_path, index_col=0, parse_dates=[0])
        df.index.name = "Date"
    else:
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
            save_name="master_df.csv",
        )


    # 2) Define columnas (igual que tu pipeline)
    # HMM SOLO con precio+volatilidad (regímenes de mercado)
    hmm_cols = ["ret", "vol_20", "mom_5", "vix"]

    # RL con contexto (macro + noticias densas)
    # Si ya creaste sentiment_7d y news_7d en tu dataset, usa esas; si no, usa sentiment/news_count.
    if "sentiment_7d" in df.columns and "news_7d" in df.columns:
        news_cols = ["sentiment_7d", "news_7d"]
    else:
        news_cols = ["sentiment", "news_count"]

    base_cols_rl = ["ret", "vol_20", "mom_5"] + settings.wb_indicators + ["vix"] + news_cols

    df = df.dropna(subset=base_cols_rl).copy()

    # 3) Define ventanas walk-forward (no cambia tu pipeline, solo evalúa)
    # Ejemplos útiles: evaluar 2020 (crash) y 2022 (bear)
    splits = [
        ("2018-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
        ("2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        ("2018-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        ("2018-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
        ("2018-01-01", "2024-12-31", "2025-01-01", "2025-12-01"),
    ]

    windows = make_windows(df, splits)
    if not windows:
        raise RuntimeError("No se generaron ventanas. Revisa rango de fechas o tamaño de datos.")

    results_rows = []

    # 4) Ejecutar cada ventana
    for w in windows:
        tr0, tr1, te0, te1 = w["train_start"], w["train_end"], w["test_start"], w["test_end"]
        train_df = w["train_df"]
        test_df = w["test_df"]

        print(f"\n=== Ventana ===")
        print(f"Train: {tr0} -> {tr1}   (n={len(train_df)})")
        print(f"Test : {te0} -> {te1}   (n={len(test_df)})")

        # 4.1) HMM entrenado SOLO en train
        hmm_res = fit_hmm_regimes(train_df, feature_cols=hmm_cols, n_states=settings.hmm_states, seed=settings.hmm_seed)

        # Aplicar HMM al train y test:
        # truco simple: refitear HMM en train y luego predecir para ambos con el mismo scaler/model
        # (usamos el modelo ya entrenado + scaler del HMM)
        # Para no tocar tu módulo, hacemos predicción inline aquí:

        from sklearn.preprocessing import StandardScaler
        import numpy as np

        hmm_model = hmm_res.model
        hmm_scaler = hmm_res.scaler

        def add_hmm_probs(df_part: pd.DataFrame) -> pd.DataFrame:
            X = hmm_scaler.transform(df_part[hmm_cols].values)
            post = hmm_model.predict_proba(X)
            states = hmm_model.predict(X)
            out = df_part.copy()
            out["hmm_state"] = states
            for k in range(settings.hmm_states):
                out[f"hmm_p{k}"] = post[:, k]
            return out

        train_hmm = add_hmm_probs(train_df)
        test_hmm = add_hmm_probs(test_df)

        # 4.2) Escalado para RL (solo features RL)
        scaled = scale_columns(train_hmm, test_hmm, cols=base_cols_rl)
        tr_s, te_s = scaled.train_df, scaled.test_df

        # Observaciones
        obs_nohmm = base_cols_rl
        obs_hmm = base_cols_rl + [f"hmm_p{k}" for k in range(settings.hmm_states)]

        # 4.3) Baseline Buy&Hold (en test)
        bh_eq = (te_s["close"] / te_s["close"].iloc[0]).rename("equity")
        bh_metrics = summarize_equity(bh_eq)

        # 4.4) Entrenar DQN con y sin HMM
        model_hmm = train_dqn(tr_s, obs_hmm, timesteps=settings.rl_timesteps, seed=settings.rl_seed, fee=settings.rl_fee)
        model_no = train_dqn(tr_s, obs_nohmm, timesteps=settings.rl_timesteps, seed=settings.rl_seed, fee=settings.rl_fee)

        ev_hmm = evaluate(model_hmm, te_s, obs_hmm, fee=settings.rl_fee)
        ev_no = evaluate(model_no, te_s, obs_nohmm, fee=settings.rl_fee)

        row = {
            "train_start": tr0, "train_end": tr1,
            "test_start": te0, "test_end": te1,
            "bh_cumret": bh_metrics["Cumulative Return"],
            "bh_sharpe": bh_metrics["Sharpe"],
            "bh_maxdd": bh_metrics["Max Drawdown"],
            "dqn_hmm_cumret": ev_hmm["metrics"]["Cumulative Return"],
            "dqn_hmm_sharpe": ev_hmm["metrics"]["Sharpe"],
            "dqn_hmm_maxdd": ev_hmm["metrics"]["Max Drawdown"],
            "dqn_no_cumret": ev_no["metrics"]["Cumulative Return"],
            "dqn_no_sharpe": ev_no["metrics"]["Sharpe"],
            "dqn_no_maxdd": ev_no["metrics"]["Max Drawdown"],
        }
        results_rows.append(row)

        c_hmm, _ = action_counts(model_hmm, te_s, obs_hmm, fee=settings.rl_fee)
        c_no, _ = action_counts(model_no, te_s, obs_nohmm, fee=settings.rl_fee)
        print("Acciones DQN+HMM:", c_hmm)
        print("Acciones DQN no  :", c_no)
        
        print("Buy&Hold:", bh_metrics)
        print("DQN+HMM :", ev_hmm["metrics"])
        print("DQN no  :", ev_no["metrics"])

    # 5) Guardar resultados
    out_dir = Path(settings.artifacts_dir) / "walkforward"
    ensure_dir(str(out_dir))
    res_df = pd.DataFrame(results_rows)
    save_csv(res_df, str(out_dir / "walkforward_metrics.csv"), index=False)

    print(f"\nGuardado: {out_dir / 'walkforward_metrics.csv'}")


if __name__ == "__main__":
    main()
