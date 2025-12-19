import pandas as pd
import requests

def load_bcrp_api(series_code: str, start_period: str, end_period: str, lang: str="esp") -> pd.DataFrame:
    url = f"https://estadisticas.bcrp.gob.pe/estadisticas/series/api/{series_code}/json/{start_period}/{end_period}/{lang}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    # Parse flexible: buscamos pares (fecha/periodo, valor)
    rows = []
    def walk(obj):
        if isinstance(obj, dict):
            # muchos JSON del BCRP traen "periodo"/"fecha" y "valor"
            keys = {k.lower(): k for k in obj.keys()}
            if ("valor" in keys) and (("periodo" in keys) or ("fecha" in keys)):
                dkey = keys.get("fecha") or keys.get("periodo")
                rows.append((obj.get(dkey), obj.get(keys["valor"])))
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for it in obj:
                walk(it)

    walk(data)
    df = pd.DataFrame(rows, columns=["date", series_code]).dropna()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[series_code] = pd.to_numeric(df[series_code], errors="coerce")
    df = df.dropna().set_index("date").sort_index()
    return df
