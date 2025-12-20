import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from src.config import settings
from src.data.build_dataset import build_dataset

def main():
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

    print("\n=== Columnas ===")
    print(df.columns.tolist())

    print("\n=== Describe sentiment ===")
    print(df["sentiment"].describe())

    non_zero = (df["sentiment"].fillna(0) != 0).sum()
    print(f"\nDías sentiment != 0: {non_zero} de {len(df)}")

    print("\nTop 10 valores abs(sentiment) más grandes:")
    print(df["sentiment"].abs().sort_values(ascending=False).head(10))

    print(df[["sentiment","news_count"]].tail(10))
    
if __name__ == "__main__":
    main()