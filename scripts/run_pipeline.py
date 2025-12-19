import pandas as pd

from src.config import settings
from src.data.yahoo import load_yahoo_close
from data.vix_csv import load_vix_csv
from src.data.bcrp import load_bcrp_api
from src.data.alphavantage_news import fetch_news_sentiment, news_to_daily_sentiment
from src.features.features import add_price_features, build_master_df
from src.hmm.hmm_regime import fit_hmm
from src.rl.train import train_dqn, run_policy, summarize
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

def main():
    if not settings.av_api_key:
        raise RuntimeError("Falta ALPHAVANTAGE_API_KEY en tu entorno (.env).")

    # 1) Precios EPU
    price = load_yahoo_close(settings.ticker, settings.start, settings.end)

    # 2) Macro Perú (ejemplos; ajusta series y rango)
    # Estructura BCRPData GET :contentReference[oaicite:16]{index=16}
    tc = load_bcrp_api("PN01207PM", "2018-1", "2025-12")     # tipo de cambio mensual (ejemplo)
    ipc = load_bcrp_api("PN01271PM", "2018-1", "2025-12")    # IPC mensual (ejemplo)
    macro = tc.join(ipc, how="outer")

    # 3) CSV vol
    vol = load_vix_csv("data/raw/volatilidad_vix.csv")

    # 4) Alpha Vantage News & Sentiment (ticker=EPU, topics opcional) :contentReference[oaicite:17]{index=17}
    payload = fetch_news_sentiment(
        base_url=settings.av_base_url,
        api_key=settings.av_api_key,
        tickers=settings.ticker,
        topics=settings.av_topics,
        time_from=pd.Timestamp(settings.start),
        time_to=pd.Timestamp(settings.end),
        sort=settings.av_sort,
        limit=settings.av_limit,
    )
    sent = news_to_daily_sentiment(payload, target_ticker=settings.ticker)

    # Merge + features
    price_f = add_price_features(price)
    df = build_master_df(price_f, macro, vol, sent)

    # Features base (sin HMM)
    base_cols = ["ret","vol_20","mom_5","PN01207PM","PN01271PM","vol","sentiment"]
    df = df.dropna(subset=base_cols)

    # Split temporal
    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split].copy(), df.iloc[split:].copy()

    # HMM
    df_hmm, hmm, scaler = fit_hmm(df, base_cols, n_states=3)
    train_hmm, test_hmm = df_hmm.iloc[:split].copy(), df_hmm.iloc[split:].copy()

    hmm_cols = base_cols + ["hmm_p0","hmm_p1","hmm_p2"]
    nohmm_cols = base_cols

    # RL: con HMM
    model_hmm = train_dqn(train_hmm, hmm_cols, timesteps=50_000)
    eq_hmm = run_policy(model_hmm, test_hmm, hmm_cols)

    # RL: sin HMM
    model_no = train_dqn(train_df, nohmm_cols, timesteps=50_000)
    eq_no = run_policy(model_no, test_df, nohmm_cols)

    print("Resultados (test):")
    print(pd.DataFrame({"DQN+HMM": summarize(eq_hmm), "DQN sin HMM": summarize(eq_no)}).T)

if __name__ == "__main__":
    main()
