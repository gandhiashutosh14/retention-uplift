"""
Stage 3: what the historical discount flag says about the discount's effect.

Three estimators, each with its own failure mode, run on the same rows:
  segment difference   churn(no discount) - churn(discount) per segment with Wilson bounds
  IPW-adjusted ATE     inverse-propensity weighting inside the segments that have treated rows,
                       with a bootstrap interval; corrects observed confounding on the features
  T-learner            two calibrated churn models (treated / untreated) on the common-support
                       segments; tau(x) = p0(x) - p1(x); scored by Qini on out-of-fold predictions

All three are reported; none is used blindly. The policy stage consumes the *interval*, and
treats the save rate as a scenario parameter when the interval includes zero.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .audit import segment_table
from .data import Dataset, encode, feature_columns


def qini_coefficient(tau: np.ndarray, treated: np.ndarray, y: np.ndarray) -> float:
    """Normalised Qini coefficient from scikit-uplift (area between the model's Qini curve and the random
    line, relative to the perfect curve). Uplift here is churn *reduction*, so the response is (1 - churn).
    Positive means the ranking finds customers whom the discount retains; about zero means no signal."""
    from sklift.metrics import qini_auc_score
    return float(qini_auc_score(1 - y, tau, treated))


def class_transformation_qini(df: pd.DataFrame, reference: pd.DataFrame, *, seed: int = 0) -> Dict[str, Any]:
    """scikit-uplift's class-transformation learner on the same rows, out of fold. Its assumption
    (balanced random treatment) is violated here, so it is a reference point, not an estimator."""
    from sklift.models import ClassTransformation
    cols = feature_columns(reference)
    X = encode(df, reference, cols)
    t, y = df.discount.to_numpy(), 1 - df.churn.to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    tau = np.zeros(len(df))
    for tr_idx, te_idx in cv.split(X, t * 2 + y):
        m = ClassTransformation(HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_leaf_nodes=7, random_state=seed))
        m.fit(X.iloc[tr_idx], y[tr_idx], t[tr_idx])
        tau[te_idx] = m.predict(X.iloc[te_idx])
    from sklift.metrics import qini_auc_score
    return {"qini": float(qini_auc_score(y, tau, t)), "share_positive": float((tau > 0).mean())}


def ipw_ate(df: pd.DataFrame, reference: pd.DataFrame, *, n_boot: int = 300, seed: int = 0) -> Dict[str, Any]:
    """Churn reduction from the discount, IPW-adjusted, within the rows given (common support)."""
    cols = [c for c in feature_columns(reference) if c not in ("estimated_clv",)]
    X = encode(df, reference, cols).to_numpy(dtype=float)
    t, y = df.discount.to_numpy(), df.churn.to_numpy()
    prop = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    e = np.clip(cross_val_predict(prop, X, t, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1], 0.02, 0.98)
    def ate(idx):
        tt, yy, ee = t[idx], y[idx], e[idx]
        w1, w0 = tt / ee, (1 - tt) / (1 - ee)
        return float(np.sum(w0 * yy) / np.sum(w0) - np.sum(w1 * yy) / np.sum(w1))   # churn(untreated) - churn(treated)
    rng = np.random.default_rng(seed)
    point = ate(np.arange(len(df)))
    boots = [ate(rng.integers(0, len(df), len(df))) for _ in range(n_boot)]
    return {"rows": int(len(df)), "treated": int(t.sum()), "ate_churn_reduction": point,
            "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5)),
            "propensity_range": [float(e.min()), float(e.max())]}


def t_learner(df: pd.DataFrame, reference: pd.DataFrame, *, seed: int = 0) -> Dict[str, Any]:
    cols = feature_columns(reference)
    X = encode(df, reference, cols)
    t, y = df.discount.to_numpy(), df.churn.to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    tau = np.zeros(len(df))
    for train_idx, test_idx in cv.split(X, t * 2 + y):
        m0 = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=7, random_state=seed)
        m1 = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_leaf_nodes=3, l2_regularization=5.0, random_state=seed)
        tr0, tr1 = train_idx[t[train_idx] == 0], train_idx[t[train_idx] == 1]
        m0.fit(X.iloc[tr0], y[tr0])
        m1.fit(X.iloc[tr1], y[tr1])
        tau[test_idx] = m0.predict_proba(X.iloc[test_idx])[:, 1] - m1.predict_proba(X.iloc[test_idx])[:, 1]
    return {"rows": int(len(df)), "tau_mean": float(tau.mean()), "tau_p10": float(np.percentile(tau, 10)),
            "tau_p90": float(np.percentile(tau, 90)), "qini": qini_coefficient(tau, t, y),
            "share_positive": float((tau > 0).mean())}


def run_uplift(ds: Dataset) -> Dict[str, Any]:
    tr = ds.train
    segs = segment_table(tr)
    support = [s["segment"] for s in segs if s["n_treated"] > 0 and s["churn_control"] not in (0.0, 1.0)]
    rows = tr[tr.segment.isin(support)]
    out = {"segments": segs, "common_support_segments": support,
           "ipw": ipw_ate(rows, tr) if len(rows) else None,
           "t_learner": t_learner(rows, tr) if len(rows) else None,
           "class_transformation": class_transformation_qini(rows, tr) if len(rows) else None}
    ipw = out["ipw"]
    out["conclusion"] = (
        f"Within the {len(rows)} customers in segments where the discount was tried on people who could go either way "
        f"({', '.join(support)}), the IPW-adjusted churn reduction is {ipw['ate_churn_reduction']:+.3f} with a 95% bootstrap "
        f"interval [{ipw['ci_low']:+.3f}, {ipw['ci_high']:+.3f}]. "
        + ("The interval includes zero: the data does not demonstrate that the discount saves anyone. "
           if ipw["ci_low"] <= 0 <= ipw["ci_high"] else "The interval excludes zero. ")
        + f"Out-of-fold normalised Qini coefficients: T-learner {out['t_learner']['qini']:+.3f}, class transformation "
        f"{out['class_transformation']['qini']:+.3f} (zero is random ordering, one is perfect). "
        "The policy therefore treats the save rate as a scenario parameter and the campaign as the experiment that measures it."
    )
    return out
