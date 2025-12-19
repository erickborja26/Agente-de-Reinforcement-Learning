import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

def fit_hmm(df: pd.DataFrame, feature_cols: list[str], n_states: int = 3, seed: int = 7):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])

    hmm = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=500, random_state=seed)
    hmm.fit(X)

    post = hmm.predict_proba(X)
    out = df.copy()
    out["hmm_state"] = hmm.predict(X)
    for k in range(n_states):
        out[f"hmm_p{k}"] = post[:, k]

    return out, hmm, scaler
