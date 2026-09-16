"""
Stage 4: who gets the 50 offers.

Expected net gain of offering customer i, under save rate s for persuadable customers:

    gain_i(s) = s * relevance_i * p_i * CLV_i - 20

p_i is the calibrated no-offer churn probability, CLV_i the value lost if they churn, relevance_i
a segment multiplier (0 for sure things and lost causes, 1 for pricing complaints, a configurable
fraction for support complaints, since a price discount does not answer a support complaint).
The knapsack with equal costs is solved exactly by sorting: offer to the highest gains while the
gain is positive and the budget holds. Because s multiplies every candidate's benefit, the *order*
does not depend on s; only how many customers clear the $20 bar does. That is why the decision
is reported across save rates instead of pretending to know one.

A second list reserves a randomised control inside the top pool so the campaign measures s.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .data import BUDGET, DISCOUNT


@dataclass
class PolicyConfig:
    budget: float = BUDGET
    discount: float = DISCOUNT
    save_rates: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.5, 1.0])
    primary_save_rate: float = 0.3
    relevance: Dict[str, float] = field(default_factory=lambda: {
        "pricing/neutral": 1.0, "pricing/negative": 1.0, "customer_support/negative": 0.5,
        "service_quality/positive": 0.0, "churn_intent/negative": 0.0})
    control_size: int = 10          # for the learn-while-earning list
    seed: int = 0

    @property
    def max_offers(self) -> int:
        return int(self.budget // self.discount)


def score_candidates(test: pd.DataFrame, p_churn: np.ndarray, cfg: PolicyConfig) -> pd.DataFrame:
    df = test[["customer_id", "segment", "estimated_clv", "monthly_revenue", "estimated_remaining_months"]].copy()
    df["p_churn"] = p_churn
    df["relevance"] = df.segment.map(cfg.relevance).fillna(0.0)
    df["value_at_risk"] = df.p_churn * df.estimated_clv
    df["benefit_per_unit_save_rate"] = df.relevance * df.value_at_risk     # gain(s) = s * this - discount
    df["breakeven_save_rate"] = np.where(df.benefit_per_unit_save_rate > 0, cfg.discount / df.benefit_per_unit_save_rate, np.inf)
    df["rank"] = df.benefit_per_unit_save_rate.rank(ascending=False, method="first").astype(int)
    return df.sort_values("rank").reset_index(drop=True)


def select(scored: pd.DataFrame, cfg: PolicyConfig, save_rate: float) -> pd.DataFrame:
    """Exact knapsack for equal costs: best gains first, stop at the budget or when gains stop being positive."""
    gain = save_rate * scored.benefit_per_unit_save_rate - cfg.discount
    chosen = scored[gain > 0].head(cfg.max_offers).copy()
    chosen["expected_gain"] = gain[chosen.index]
    chosen["save_rate"] = save_rate
    return chosen


def scenario_table(scored: pd.DataFrame, cfg: PolicyConfig) -> List[Dict[str, Any]]:
    rows = []
    for s in cfg.save_rates:
        sel = select(scored, cfg, s)
        rows.append({"save_rate": s, "offers": int(len(sel)), "spend": float(len(sel) * cfg.discount),
                     "expected_net_gain": float(sel.expected_gain.sum()) if len(sel) else 0.0,
                     "expected_value_saved": float((s * sel.benefit_per_unit_save_rate).sum()) if len(sel) else 0.0,
                     "segments": sel.segment.value_counts().to_dict() if len(sel) else {}})
    return rows


def learn_while_earning(scored: pd.DataFrame, cfg: PolicyConfig, save_rate: float) -> Dict[str, Any]:
    """Take the customers the pure policy would offer to, and hold `control_size` of them out at random.
    Offer and control then come from the same pool, so the difference in their realised churn estimates the
    save rate for the next campaign. The control customers cost nothing; the price is the expected gain of
    the offers not sent to them."""
    pool = select(scored, cfg, save_rate).copy()
    rng = np.random.default_rng(cfg.seed)
    control_idx = rng.choice(pool.index, size=min(cfg.control_size, len(pool)), replace=False)
    pool["arm"] = np.where(pool.index.isin(control_idx), "control", "offer")
    offers, control = pool[pool.arm == "offer"], pool[pool.arm == "control"]
    return {"pool_size": int(len(pool)), "offers": int(len(offers)), "control": int(len(control)),
            "expected_gain_forgone_vs_pure_exploitation": float(control.expected_gain.sum()), "table": pool}
