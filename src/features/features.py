import pandas as pd
import numpy as np

def add_price_features(price: pd.DataFrame) -> pd.DataFrame:
    df = price.copy()
    df["ret"] = np.log(df["close"]).diff()
    df["vol_20"] = df["ret"].rolling(20).std() * np.sqrt(252)
    df["mom_5"] = df["ret"].rolling(5).sum()
    return df

def build_master_df(price_f, macro_df, vol_df, sent_df) -> pd.DataFrame:
    # Igualar a frecuencia diaria
    macro_d = macro_df.resample("D").ffill()
    vol_d = vol_df.resample("D").ffill()
    sent_d = sent_df.resample("D").ffill()

    df = (price_f
          .join(macro_d, how="left")
          .join(vol_d, how="left")
          .join(sent_d, how="left"))

    df = df.replace([np.inf, -np.inf], np.nan).ffill().dropna()
    return df
