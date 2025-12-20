import time
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import requests

def _to_av_time(ts: pd.Timestamp) -> str:
    # YYYYMMDDTHHMM
    return ts.strftime("%Y%m%dT%H%M")

def _parse_yyyymmdd(s: str) -> pd.Timestamp:
    # s = 'YYYYMMDD'
    y = int(s[0:4]); m = int(s[4:6]); d = int(s[6:8])
    return pd.Timestamp(year=y, month=m, day=d)

def fetch_news_sentiment_payload(
    base_url: str,
    api_key: str,
    ticker: str,
    time_from: pd.Timestamp,
    time_to: pd.Timestamp,
    topics: Optional[str],
    sort: str,
    limit: int,
    timeout: int = 60
) -> Dict:
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "apikey": api_key,
        "time_from": _to_av_time(time_from),
        "time_to": _to_av_time(time_to),
        "sort": sort,
        "limit": str(limit),
    }
    if ticker:
        params["tickers"] = ticker
    if topics:
        params["topics"] = topics

    r = requests.get(base_url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def payload_to_daily_sentiment(payload: Dict, ticker: str) -> pd.DataFrame:
    """
    Devuelve DF diario: index=date, column=sentiment.
    Usa ticker_sentiment_score (ponderado por relevance_score) si existe; sino overall_sentiment_score.
    """
    feed = payload.get("feed", [])
    rows = []
    for item in feed:
        tp = item.get("time_published") or item.get("time") or item.get("time_published_utc")
        if not tp:
            continue
        try:
            day = _parse_yyyymmdd(tp[:8])
        except Exception:
            continue

        score = None
        ts = item.get("ticker_sentiment", [])
        if isinstance(ts, list):
            for t in ts:
                if str(t.get("ticker", "")).upper() == ticker.upper():
                    rel = float(t.get("relevance_score", 1.0) or 1.0)
                    sc = float(t.get("ticker_sentiment_score", 0.0) or 0.0)
                    score = sc * rel
                    break
        if score is None:
            score = float(item.get("overall_sentiment_score", 0.0) or 0.0)

        rows.append((day, score))

    if not rows:
        return pd.DataFrame({"sentiment": []}, index=pd.to_datetime([]))

    df = pd.DataFrame(rows, columns=["date", "sentiment"]).dropna()
    df = df.set_index("date").sort_index()
    
    daily = df.groupby(level=0).agg(
        sentiment=("sentiment", "mean"),
        news_count=("sentiment", "size")
    )
    return daily

def fetch_daily_sentiment_with_cache(
    base_url: str,
    api_key: str,
    ticker: str,
    start: str,
    end: str,
    topics: Optional[str],
    sort: str,
    limit: int,
    sleep_sec: float,
    cache_dir: str
) -> pd.DataFrame:
    """
    Descarga por chunks (anuales) y cachea un CSV diario.
    Si hay rate-limit (Note/Information/Error Message), devuelve lo acumulado o vacío.
    """
    cache_path = Path(cache_dir) / f"news_sentiment_{ticker}_{start}_{end}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["date"]).set_index("date").sort_index()

    if not api_key:
        # sin API key => sentimiento neutral vacío (lo llenará a 0 más adelante)
        return pd.DataFrame({"sentiment": []}, index=pd.to_datetime([]))

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    all_daily = []

    for year in range(start_ts.year, end_ts.year + 1):
        y0 = pd.Timestamp(f"{year}-01-01")
        y1 = pd.Timestamp(f"{year}-12-31 23:59")
        tf = max(start_ts, y0)
        tt = min(end_ts, y1)
        if tf > tt:
            continue

        payload = fetch_news_sentiment_payload(
            base_url=base_url,
            api_key=api_key,
            ticker=ticker,
            time_from=tf,
            time_to=tt,
            topics=topics,
            sort=sort,
            limit=limit,
        )

        # Rate-limit / errores
        if any(k in payload for k in ["Note", "Information", "Error Message"]):
            msg = payload.get("Note") or payload.get("Information") or payload.get("Error Message")
            print(f"[AlphaVantage] Aviso/Error: {msg}")
            break

        daily = payload_to_daily_sentiment(payload, ticker)
        all_daily.append(daily)

        time.sleep(sleep_sec)

    if not all_daily:
        out = pd.DataFrame({"sentiment": []}, index=pd.to_datetime([]))
    else:
        out = pd.concat(all_daily, axis=0).groupby(level=0).mean().sort_index()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.reset_index().rename(columns={"index": "date"}).to_csv(cache_path, index=False)
    return out
