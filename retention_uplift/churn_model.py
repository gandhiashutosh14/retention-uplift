"""
Stage 2: calibrated churn propensity P(churn | x, no discount).

Trained on the untreated rows only, so the probability is the no-offer counterfactual the policy
needs; treated rows are too few (135) and non-random to model jointly. Calibration matters more
than ranking here because the policy multiplies the probability by money.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .data import Dataset, encode, feature_columns


@dataclass
class ChurnModel:
    columns: List[str]
    model: Any
    metrics: Dict[str, float]

    def predict(self, df: pd.DataFrame, reference: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(encode(df, reference, self.columns))[:, 1]


def _base():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=0)


def fit_churn_model(ds: Dataset, *, drop_feedback: bool = False) -> ChurnModel:
    tr = ds.train[ds.train.discount == 0]
    cols = feature_columns(ds.train)
    if drop_feedback:
        cols = [c for c in cols if c not in ("feedback_category", "sentiment", "complaint_intensity")]
    X, y = encode(tr, ds.train, cols), tr.churn.to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    raw = cross_val_predict(_base(), X, y, cv=cv, method="predict_proba")[:, 1]
    model = CalibratedClassifierCV(_base(), method="isotonic", cv=cv)
    model.fit(X, y)
    calibrated_cv = cross_val_predict(CalibratedClassifierCV(_base(), method="isotonic", cv=3), X, y, cv=cv, method="predict_proba")[:, 1]
    metrics = {"train_rows_untreated": int(len(tr)),
               "cv_auc_raw": float(roc_auc_score(y, raw)), "cv_brier_raw": float(brier_score_loss(y, raw)),
               "cv_auc_calibrated": float(roc_auc_score(y, calibrated_cv)), "cv_brier_calibrated": float(brier_score_loss(y, calibrated_cv))}
    if "churn" in ds.test.columns:      # the test label is used for scoring only, never for fitting
        pt = model.predict_proba(encode(ds.test, ds.train, cols))[:, 1]
        metrics.update({"test_auc": float(roc_auc_score(ds.test.churn, pt)),
                        "test_brier": float(brier_score_loss(ds.test.churn, pt)),
                        "test_mean_predicted": float(pt.mean()), "test_churn_rate": float(ds.test.churn.mean())})
    return ChurnModel(columns=cols, model=model, metrics=metrics)


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> List[Dict[str, float]]:
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum():
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "mean_predicted": float(p[m].mean()), "observed": float(y[m].mean())})
    return rows
