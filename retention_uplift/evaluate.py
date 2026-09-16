"""
Stage 5: score the campaign list on the test set.

The test set carries the churn label without any discount, so for any list of 50 customers we
can see which of them would have churned and how much CLV that is. Two scorers are reported:

  naive       counts every reached churner's CLV as saveable at rate s. It rewards targeting
              customers who churn no matter what, so it is shown but not used to choose.
  consistent  applies the same persuadability rule as the policy: a reached churner's CLV counts
              at s x relevance(segment), so lost causes (relevance 0) and support complaints
              (relevance 0.5) are weighted as the policy weights them.

    NFG(s) = s * (weighted CLV of targeted customers who churned) - 20 * (customers targeted)

Baselines are the lists the case describes (churn risk alone, value alone, risk x value, random)
plus an oracle that uses the test labels: the ceiling nobody can reach without knowing the future.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .data import DISCOUNT


def reach(test: pd.DataFrame, ids: List[str], relevance: Dict[str, float]) -> Dict[str, float]:
    sel = test[test.customer_id.isin(ids)]
    churners = sel[sel.churn == 1]
    weighted = float((churners.estimated_clv * churners.segment.map(relevance).fillna(0.0)).sum())
    spend = float(len(sel) * DISCOUNT)
    return {"targeted": int(len(sel)), "churners_reached": int(len(churners)),
            "value_reached_naive": float(churners.estimated_clv.sum()),
            "value_reached_consistent": weighted, "spend": spend,
            "breakeven_save_rate": spend / weighted if weighted > 0 else float("inf"),
            "lost_causes_targeted": int((sel.segment.map(relevance).fillna(0.0) == 0).sum())}


def nfg(r: Dict[str, float], s: float, scorer: str = "consistent") -> float:
    return s * r[f"value_reached_{scorer}"] - r["spend"]


def baselines(test: pd.DataFrame, p_churn: np.ndarray, n: int, relevance: Dict[str, float]) -> Dict[str, List[str]]:
    df = test.assign(p=p_churn, rel=test.segment.map(relevance).fillna(0.0))
    return {
        "churn-risk top-n": df.nlargest(n, "p").customer_id.tolist(),
        "clv top-n": df.nlargest(n, "estimated_clv").customer_id.tolist(),
        "risk x clv top-n (no segment logic)": df.assign(v=df.p * df.estimated_clv).nlargest(n, "v").customer_id.tolist(),
        "oracle: persuadable churners with highest weighted clv": df[df.churn == 1].assign(v=df.estimated_clv * df.rel).nlargest(n, "v").customer_id.tolist(),
        "oracle (naive): any churners with highest clv": df[df.churn == 1].nlargest(n, "estimated_clv").customer_id.tolist(),
    }


def evaluate(test: pd.DataFrame, p_churn: np.ndarray, policy_ids: List[str], save_rates: List[float],
             n: int, relevance: Dict[str, float], *, random_seeds: int = 200) -> Dict[str, Any]:
    rows = {"policy (segment-aware expected gain)": reach(test, policy_ids, relevance)}
    for name, ids in baselines(test, p_churn, n, relevance).items():
        rows[name] = reach(test, ids, relevance)
    rs = [reach(test, test.sample(n, random_state=k).customer_id.tolist(), relevance) for k in range(random_seeds)]
    rows[f"random n (mean of {random_seeds} draws)"] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
    table = []
    for name, r in rows.items():
        table.append({"list": name, **r,
                      **{f"nfg_consistent@s={s:g}": nfg(r, s, "consistent") for s in save_rates},
                      **{f"nfg_naive@s={s:g}": nfg(r, s, "naive") for s in save_rates}})
    return {"n": n, "relevance": relevance, "rows": table}
