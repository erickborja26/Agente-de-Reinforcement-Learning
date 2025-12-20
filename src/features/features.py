import pandas as pd
import numpy as np

def add_price_features(price: pd.DataFrame) -> pd.DataFrame:
    df = price.copy()
    df["ret"] = np.log(df["close"]).diff()
    df["vol_20"] = df["ret"].rolling(20).std() * np.sqrt(252)
    df["mom_5"] = df["ret"].rolling(5).sum()
    return df

def build_master_df(price_f: pd.DataFrame,
                    macro_df: pd.DataFrame,
                    vix_df: pd.DataFrame,
                    sent_df: pd.DataFrame) -> pd.DataFrame:
    macro_d = macro_df.resample("D").ffill() if macro_df is not None and not macro_df.empty else pd.DataFrame(index=price_f.index)
    vix_d = vix_df.resample("D").ffill() if vix_df is not None and not vix_df.empty else pd.DataFrame(index=price_f.index)
    sent_d = sent_df.resample("D").ffill() if sent_df is not None and not sent_df.empty else pd.DataFrame(index=price_f.index)

    df = price_f.join(macro_d, how="left").join(vix_d, how="left").join(sent_d, how="left")
    df = df.sort_index()
    return df
