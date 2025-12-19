import pandas as pd
import numpy as np
import requests
from datetime import datetime

def _to_av_time(dt: pd.Timestamp) -> str:
    # Alpha Vantage usa YYYYMMDDTHHMM :contentReference[oaicite:6]{index=6}
    return dt.strftime("%Y%m%dT%H%M")

def fetch_news_sentiment(
    base_url: str,
    api_key: str,
    tickers: str,
    topics: str | None,
    time_from: pd.Timestamp | None,
    time_to: pd.Timestamp | None,
    sort: str = "LATEST",
    limit: int = 50,
    timeout: int = 30,
) -> dict:
    params = {
        "function": "NEWS_SENTIMENT",     # :contentReference[oaicite:7]{index=7}
        "tickers": tickers,               # :contentReference[oaicite:8]{index=8}
        "sort": sort,                     # :contentReference[oaicite:9]{index=9}
        "limit": str(limit),              # :contentReference[oaicite:10]{index=10}
        "apikey": api_key,                # :contentReference[oaicite:11]{index=11}
    }
    if topics:
        params["topics"] = topics         # :contentReference[oaicite:12]{index=12}
    if time_from is not None:
        params["time_from"] = _to_av_time(time_from)  # :contentReference[oaicite:13]{index=13}
    if time_to is not None:
        params["time_to"] = _to_av_time(time_to)      # :contentReference[oaicite:14]{index=14}

    r = requests.get(base_url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def news_to_daily_sentiment(payload: dict, target_ticker: str) -> pd.DataFrame:
    """
    Convierte la respuesta JSON a un DataFrame diario con una columna 'sentiment'.
    Parser robusto: usa ticker_sentiment si existe; si no, usa overall_sentiment_score si existe.
    """
    feed = payload.get("feed", [])  # si no existe, quedará vacío
    rows = []
    for item in feed:
        # time_published suele venir en formato tipo 20240101T123000...
        tp = item.get("time_published") or item.get("time_published_utc") or item.get("time")
        if not tp:
            continue

        # Parse flexible: tomamos YYYYMMDD
        try:
            day = datetime.strptime(tp[:8], "%Y%m%d").date()
        except Exception:
            continue

        score = None
        # Preferir sentimiento específico del ticker si la API lo incluye
        ts = item.get("ticker_sentiment", [])
        if isinstance(ts, list):
            for t in ts:
                if str(t.get("ticker", "")).upper() == target_ticker.upper():
                    # weighted by relevance if present
                    rel = float(t.get("relevance_score", 1.0) or 1.0)
                    sc = float(t.get("ticker_sentiment_score", 0.0) or 0.0)
                    score = sc * rel
                    break

        if score is None:
            # fallback: overall sentiment
            score = float(item.get("overall_sentiment_score", 0.0) or 0.0)

        rows.append((pd.Timestamp(day), score))

    if not rows:
        # Si no hay noticias (o hay rate limit), devolvemos serie vacía
        return pd.DataFrame({"sentiment": []}, index=pd.to_datetime([]))

    df = pd.DataFrame(rows, columns=["date", "sentiment"]).set_index("date").sort_index()

    # Agregación diaria: media (puedes cambiar a mediana o suma)
    daily = df.groupby(df.index).mean()
    return daily
