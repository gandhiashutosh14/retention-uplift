"""Loading, feature engineering and the segment definition shared by every stage."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "telco_churn_case"
DISCOUNT = 20.0
BUDGET = 1000.0

CATEGORICAL = ["state", "area_code", "international_plan", "voice_mail_plan", "feedback_category", "sentiment"]
EXCLUDED = {"customer_id", "split", "feedback_text", "churn", "discount"}
FEEDBACK_FIELDS = ["feedback_category", "sentiment", "complaint_intensity"]


@dataclass
class Dataset:
    train: pd.DataFrame
    test: pd.DataFrame

    @classmethod
    def load(cls, data_dir: Path = DATA_DIR) -> "Dataset":
        train = pd.read_csv(data_dir / "train_new.csv")
        test = pd.read_csv(data_dir / "test_new.csv")
        for df in (train, test):
            df["segment"] = df["feedback_category"].astype(str) + "/" + df["sentiment"].astype(str)
        return cls(train, test)


def feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in EXCLUDED and c != "segment"]


def encode(df: pd.DataFrame, reference: pd.DataFrame, columns: List[str] | None = None) -> pd.DataFrame:
    """Numeric design matrix. Categorical codes are fixed by the reference frame so train and test agree."""
    columns = columns or feature_columns(reference)
    out = pd.DataFrame(index=df.index)
    for c in columns:
        if c in CATEGORICAL:
            cats = sorted(reference[c].astype(str).unique())
            out[c] = pd.Categorical(df[c].astype(str), categories=cats).codes
        else:
            out[c] = pd.to_numeric(df[c], errors="coerce")
    return out


def value_at_risk(df: pd.DataFrame) -> np.ndarray:
    """What is lost if the customer churns: the estimated CLV column (monthly revenue x remaining months)."""
    return df["estimated_clv"].to_numpy(dtype=float)
