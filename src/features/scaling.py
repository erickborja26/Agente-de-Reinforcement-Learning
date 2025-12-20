from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler

@dataclass
class ScaleResult:
    scaler: StandardScaler
    train_df: pd.DataFrame
    test_df: pd.DataFrame

def scale_columns(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: List[str]) -> ScaleResult:
    scaler = StandardScaler()
    train_out = train_df.copy()
    test_out = test_df.copy()

    train_out[cols] = scaler.fit_transform(train_out[cols].values)
    test_out[cols] = scaler.transform(test_out[cols].values)

    return ScaleResult(scaler=scaler, train_df=train_out, test_df=test_out)
