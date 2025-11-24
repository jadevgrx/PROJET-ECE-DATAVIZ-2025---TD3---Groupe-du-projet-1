import pandas as pd
from pathlib import Path

# Dossiers (depuis app/utils.py)
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def load_df_clean():
    """Charge le dataset de transactions nettoyées."""
    path = PROCESSED_DIR / "online_retail_clean.parquet"
    df = pd.read_parquet(path)
    return df


def load_rfm():
    """Charge la table RFM (avec scores/segments)."""
    path = PROCESSED_DIR / "rfm_segments.csv"
    df = pd.read_csv(path, index_col=0)
    return df


def load_cohort_retention():
    """Charge la matrice de rétention par cohorte."""
    path = PROCESSED_DIR / "cohort_retention.csv"
    df = pd.read_csv(path, index_col=0)
    return df


def load_cohort_avg_revenue():
    """Charge le revenu moyen par âge de cohorte."""
    path = PROCESSED_DIR / "cohort_avg_revenue.csv"
    df = pd.read_csv(path, index_col=0)
    return df
