import pandas as pd
import numpy as np
from pathlib import Path

from .yahoo import load_yahoo_close
from .worldbank import fetch_world_bank_macro
from .vix_csv import load_vix_csv
from .alphavantage_news import fetch_daily_sentiment_with_cache
from ..features.features import add_price_features, build_master_df

def build_dataset(
    ticker: str,
    start: str,
    end: str,
    vix_csv_path: str,
    wb_country: str,
    wb_indicators: list[str],
    av_base_url: str,
    av_api_key: str,
    av_topics: str | None,
    av_sort: str,
    av_limit: int,
    av_sleep_sec: float,
    cache_dir: str,
    processed_dir: str,
    save_name: str = "master_df.csv"
) -> pd.DataFrame:
    """
    Devuelve DF diario unificado con:
      close + features (ret, vol_20, mom_5)
      + macro WB (anual -> ffill diario)
      + vix (diario -> ffill diario)
      + sentiment (diario -> ffill diario; si vacío => 0)
    """
    price = load_yahoo_close(ticker, start, end)
    price_f = add_price_features(price)

    macro = fetch_world_bank_macro(wb_country, wb_indicators, start, end)

    vix = load_vix_csv(vix_csv_path)

    sent = fetch_daily_sentiment_with_cache(
        base_url=av_base_url,
        api_key=av_api_key,
        ticker="",
        start=start,
        end=end,
        topics=av_topics,
        sort=av_sort,
        limit=av_limit,
        sleep_sec=av_sleep_sec,
        cache_dir=cache_dir,
    )

    df = build_master_df(price_f, macro, vix, sent)

    # Sentiment limpio (si no hay data, queda 0)
    df["sentiment"] = (
        df["sentiment"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(-1.0, 1.0)
    )

    # sentimiento neutral si no hubo data
    if "sentiment" not in df.columns:
        df["sentiment"] = 0.0
    df["sentiment"] = df["sentiment"].fillna(0.0)

    df = df.replace([np.inf, -np.inf], np.nan).ffill().dropna()
    
    # si no hay noticias => 0
    df["sentiment"] = df["sentiment"].fillna(0.0)
    df["news_count"] = df.get("news_count", 0).fillna(0)
    
    # rolling 7 días (más estable para RL)
    df["sentiment_7d"] = df["sentiment"].rolling(7, min_periods=1).mean()
    df["news_7d"] = df["news_count"].rolling(7, min_periods=1).sum()
    
    # ✅ GUARDAR DF UNIFICADO (4 fuentes)
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(processed_dir) / save_name
    df.to_csv(out_path, index=True)
    print(f"[OK] DataFrame unificado guardado en: {out_path.resolve()}")
    
    return df
