import pandas as pd
import requests

def fetch_world_bank_indicator(country: str, indicator: str, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Descarga indicador anual desde World Bank y lo devuelve como DF indexado por fecha (YYYY-01-01).
    """
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
    params = {
        "format": "json",
        "per_page": 20000,
        "date": f"{start_year}:{end_year}",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise RuntimeError(f"Respuesta WorldBank inesperada para {indicator}: {payload[:1]}")

    rows = []
    for item in payload[1]:
        year = item.get("date")
        val = item.get("value")
        if year is None or val is None:
            continue
        dt = pd.Timestamp(f"{year}-01-01")
        try:
            fv = float(val)
        except Exception:
            continue
        rows.append((dt, fv))

    df = pd.DataFrame(rows, columns=["date", indicator]).dropna()
    df = df.set_index("date").sort_index()
    return df

def fetch_world_bank_macro(country: str, indicators: list[str], start: str, end: str) -> pd.DataFrame:
    start_year = pd.Timestamp(start).year
    end_year = pd.Timestamp(end).year

    out = None
    for ind in indicators:
        df = fetch_world_bank_indicator(country, ind, start_year, end_year)
        out = df if out is None else out.join(df, how="outer")

    if out is None:
        return pd.DataFrame()

    return out.sort_index()
