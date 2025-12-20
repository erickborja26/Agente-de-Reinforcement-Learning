from dataclasses import dataclass
from typing import List, Tuple, Dict
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

@dataclass
class HMMResult:
    df: pd.DataFrame
    model: GaussianHMM
    scaler: StandardScaler
    transition: pd.DataFrame
    emission_means: pd.DataFrame
    state_labels: Dict[int, str]

def fit_hmm_regimes(df: pd.DataFrame, feature_cols: List[str], n_states: int = 3, seed: int = 7) -> HMMResult:
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].values)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=800,
        random_state=seed,
        min_covar=1e-4
    )
    model.fit(X)

    post = model.predict_proba(X)
    states = model.predict(X)

    out = df.copy()
    out["hmm_state"] = states
    for k in range(n_states):
        out[f"hmm_p{k}"] = post[:, k]

    transition = pd.DataFrame(model.transmat_)
    emission_means = pd.DataFrame(model.means_, columns=feature_cols)

    # Etiquetado simple por retorno medio (en espacio original)
    means_ret = out.groupby("hmm_state")["ret"].mean().sort_values()
    order = means_ret.index.tolist()

    labels = {}
    if len(order) == 3:
        labels[order[0]] = "bear"
        labels[order[1]] = "sideways"
        labels[order[2]] = "bull"
    else:
        for i, st in enumerate(order):
            labels[st] = f"state_{i}"

    return HMMResult(
        df=out,
        model=model,
        scaler=scaler,
        transition=transition,
        emission_means=emission_means,
        state_labels=labels
    )
