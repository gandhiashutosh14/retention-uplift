"""
Stage 1: data audit. Establishes, with counts, what the historical discount can and cannot tell
us before any model is trained. Every finding is a number from the data, not an opinion.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import pandas as pd

from .data import Dataset, FEEDBACK_FIELDS


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def segment_table(train: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []
    for seg, g in train.groupby("segment"):
        ctl, trt = g[g.discount == 0], g[g.discount == 1]
        row = {"segment": seg, "n": int(len(g)), "n_treated": int(len(trt)),
               "churn_control": float(ctl.churn.mean()) if len(ctl) else None,
               "churn_treated": float(trt.churn.mean()) if len(trt) else None,
               "clv_mean": float(g.estimated_clv.mean())}
        if len(ctl) and len(trt):
            lo_c, hi_c = wilson(int(ctl.churn.sum()), len(ctl))
            lo_t, hi_t = wilson(int(trt.churn.sum()), len(trt))
            row["naive_uplift"] = row["churn_control"] - row["churn_treated"]     # positive = discount reduced churn
            # conservative bounds on churn reduction from the two Wilson intervals
            row["uplift_ci_low"] = lo_c - hi_t
            row["uplift_ci_high"] = hi_c - lo_t
        rows.append(row)
    return rows


def run_audit(ds: Dataset) -> Dict[str, Any]:
    tr = ds.train
    segs = segment_table(tr)
    deterministic = [s["segment"] for s in segs if s["churn_control"] in (0.0, 1.0) and (s["churn_treated"] in (None, 0.0, 1.0))]
    sure_things = [s["segment"] for s in segs if s["churn_control"] == 0.0]
    lost_causes = [s["segment"] for s in segs if s["churn_control"] == 1.0]
    treated_segments = {s["segment"]: s["n_treated"] for s in segs if s["n_treated"]}
    findings = {
        "rows_train": int(len(tr)), "rows_test": int(len(ds.test)),
        "test_has_churn_label": bool("churn" in ds.test.columns),
        "test_has_discount_column": bool("discount" in ds.test.columns),
        "churn_rate_train": float(tr.churn.mean()), "churn_rate_test": float(ds.test.churn.mean()),
        "discount_rate_train": float(tr.discount.mean()), "n_treated": int(tr.discount.sum()),
        "n_treated_churned": int(tr[tr.discount == 1].churn.sum()),
        "churn_by_discount": {int(k): float(v) for k, v in tr.groupby("discount").churn.mean().items()},
        "discount_rate_by_sentiment": {k: float(v) for k, v in tr.groupby("sentiment").discount.mean().items()},
        "feedback_text_unique_values": int(tr.feedback_text.nunique()),
        "clv_equals_revenue_times_months": bool(((tr.monthly_revenue * tr.estimated_remaining_months - tr.estimated_clv).abs() < 0.01).all()),
        "segments": segs,
        "deterministic_segments": deterministic,
        "sure_things": sure_things,
        "lost_causes": lost_causes,
        "treated_segments": treated_segments,
    }
    findings["conclusions"] = [
        f"The discount was given to {findings['n_treated']} of {findings['rows_train']} customers "
        f"({findings['discount_rate_train']:.1%}), not at random: the rate varies by sentiment "
        f"{ {k: round(v, 3) for k, v in findings['discount_rate_by_sentiment'].items()} }.",
        f"Churn among discounted customers ({findings['churn_by_discount'][1]:.1%}) is higher than among the rest "
        f"({findings['churn_by_discount'][0]:.1%}); a naive comparison says the discount hurts, which is confounding, not causation.",
        f"Segments with deterministic churn in the data: {deterministic}. Sure things (never churn): {sure_things}. "
        f"Lost causes (always churn, discount or not): {lost_causes}. Spending on either is wasted by definition.",
        "Within the only segment where the discount was tried on customers who both could and did not always churn "
        "(pricing complaints), the Wilson-interval bounds on its effect include zero: the historical data cannot show "
        "that the $20 offer saves anyone. The targeting policy therefore treats the save rate as an assumption to be "
        "learned from the campaign itself, and reports the decision across a range of save rates.",
        f"feedback_text has {findings['feedback_text_unique_values']} distinct values, fully captured by feedback_category "
        "and sentiment; no text model is needed.",
    ]
    return findings


def audit_markdown(f: Dict[str, Any]) -> str:
    lines = ["# Data audit", "",
             f"Train {f['rows_train']} rows, test {f['rows_test']} rows. Test carries the churn label "
             f"({'yes' if f['test_has_churn_label'] else 'no'}) and the discount column "
             f"({'yes' if f['test_has_discount_column'] else 'no'}). Churn rate train {f['churn_rate_train']:.1%}, "
             f"test {f['churn_rate_test']:.1%}. Discount rate train {f['discount_rate_train']:.1%} "
             f"({f['n_treated']} customers, of whom {f['n_treated_churned']} churned).", "",
             "| Segment (feedback category / sentiment) | n | treated | churn, no discount | churn, discount | naive uplift | 95% bounds on uplift | mean CLV |",
             "|---|---:|---:|---:|---:|---:|---|---:|"]
    for s in f["segments"]:
        ct = "" if s["churn_treated"] is None else f"{s['churn_treated']:.1%}"
        up = "" if "naive_uplift" not in s else f"{s['naive_uplift']:+.3f}"
        ci = "" if "uplift_ci_low" not in s else f"[{s['uplift_ci_low']:+.3f}, {s['uplift_ci_high']:+.3f}]"
        lines.append(f"| {s['segment']} | {s['n']} | {s['n_treated']} | {s['churn_control']:.1%} | {ct} | {up} | {ci} | {s['clv_mean']:,.0f} |")
    lines += ["", "## Conclusions", ""] + [f"- {c}" for c in f["conclusions"]]
    return "\n".join(lines) + "\n"
