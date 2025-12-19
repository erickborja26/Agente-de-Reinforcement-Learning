from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Settings:
    # Trading
    ticker: str = "EPU"
    start: str = "2018-01-01"
    end: str = "2025-12-01"

    # Alpha Vantage News & Sentiment
    av_base_url: str = "https://www.alphavantage.co/query"
    av_api_key: str = os.getenv("ALPHAVANTAGE_API_KEY", "")
    av_topics: str = "economy_macro,economy_monetary,financial_markets"  # opcional :contentReference[oaicite:2]{index=2}
    av_limit: int = 1000  # opcional :contentReference[oaicite:3]{index=3}
    av_sort: str = "LATEST"  # opcional :contentReference[oaicite:4]{index=4}

    # Paths
    volatility_xlsx: str = "data/raw/volatilidad_vix.csv"

settings = Settings()