from dataclasses import dataclass, field
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    # Trading target
    ticker: str = "EPU"
    start: str = "2018-01-01"
    end: str = "2025-12-01"

    # Data paths
    vix_csv_path: str = "data/raw/volatilidad_vix.csv"
    cache_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    artifacts_dir: str = "artifacts"

    # Alpha Vantage
    alphavantage_api_key: str = os.getenv("ALPHAVANTAGE_API_KEY", "")
    alphavantage_base_url: str = "https://www.alphavantage.co/query"
    alphavantage_topics: Optional[str] = "economy_macro,economy_monetary,financial_markets"
    alphavantage_sort: str = "LATEST"
    alphavantage_limit: int = 1000
    alphavantage_sleep_sec: float = 12.0  # evita rate-limit en cuentas free

    # World Bank (Perú)
    wb_country: str = "PER"
    wb_indicators: List[str] = field(default_factory=lambda: [
        "FP.CPI.TOTL.ZG",  # inflación anual %
        "FR.INR.RINR",     # tasa real %
        "PA.NUS.FCRF",     # tipo cambio oficial (LCU por USD)
    ])

    # HMM
    hmm_states: int = 3
    hmm_seed: int = 7

    # RL
    rl_seed: int = 7
    rl_fee: float = 0.0005
    rl_train_ratio: float = 0.8
    rl_timesteps: int = 50_000

settings = Settings()
