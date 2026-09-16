import pathlib

import numpy as np
import pandas as pd
import pytest

from retention_uplift.audit import run_audit, wilson
from retention_uplift.churn_model import fit_churn_model
from retention_uplift.data import BUDGET, DISCOUNT, Dataset
from retention_uplift.evaluate import evaluate, reach
from retention_uplift.pipeline import Critic, run_pipeline
from retention_uplift.policy import PolicyConfig, learn_while_earning, scenario_table, score_candidates, select

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ds():
    return Dataset.load()


@pytest.fixture(scope="module")
def churn(ds):
    return fit_churn_model(ds)


def test_audit_finds_the_structure_of_the_data(ds):
    a = run_audit(ds)
    assert a["sure_things"] == ["service_quality/positive"] and a["lost_causes"] == ["churn_intent/negative"]
    assert a["n_treated"] == 135 and a["test_has_churn_label"] and not a["test_has_discount_column"]
    pricing = next(s for s in a["segments"] if s["segment"] == "pricing/neutral")
    assert pricing["uplift_ci_low"] < 0 < pricing["uplift_ci_high"]          # no demonstrated effect
    lo, hi = wilson(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8


def test_churn_model_is_fit_on_untreated_rows_and_scores_the_test_set(ds, churn):
    m = churn.metrics
    assert m["train_rows_untreated"] == int((ds.train.discount == 0).sum())
    assert m["test_auc"] > 0.95 and m["test_brier"] < 0.02
    p = churn.predict(ds.test, ds.train)
    assert len(p) == len(ds.test) and 0 <= p.min() and p.max() <= 1


def test_policy_respects_budget_bans_and_positive_gain(ds, churn):
    cfg = PolicyConfig()
    scored = score_candidates(ds.test, churn.predict(ds.test, ds.train), cfg)
    for s in cfg.save_rates:
        sel = select(scored, cfg, s)
        assert len(sel) * DISCOUNT <= BUDGET and len(sel) <= cfg.max_offers
        assert not sel.customer_id.duplicated().any()
        assert (sel.expected_gain > 0).all()
        assert not sel.segment.isin(["service_quality/positive", "churn_intent/negative"]).any()
    # ranking is invariant to the save rate: any smaller selection is a prefix of any larger one
    small, big = select(scored, cfg, 0.1), select(scored, cfg, 1.0)
    assert small.customer_id.tolist() == big.customer_id.tolist()[: len(small)]
    rows = scenario_table(scored, cfg)
    assert [r["offers"] for r in rows] == sorted(r["offers"] for r in rows)


def test_select_is_the_exact_knapsack_for_equal_costs():
    cfg = PolicyConfig(budget=60.0, discount=20.0)
    scored = pd.DataFrame({"customer_id": list("abcdef"), "segment": ["pricing/neutral"] * 6,
                           "estimated_clv": [100, 900, 50, 700, 30, 400], "monthly_revenue": 1, "estimated_remaining_months": 1,
                           "p_churn": [0.9, 0.1, 0.5, 0.5, 0.9, 0.2]})
    scored["relevance"] = 1.0
    scored["value_at_risk"] = scored.p_churn * scored.estimated_clv
    scored["benefit_per_unit_save_rate"] = scored.value_at_risk
    scored["rank"] = scored.benefit_per_unit_save_rate.rank(ascending=False, method="first").astype(int)
    scored = scored.sort_values("rank").reset_index(drop=True)
    sel = select(scored, cfg, save_rate=0.5)
    # gains at s=0.5: a 25, b 25, c -7.5, d 155, e -6.5, f 20 -> best three within 3 offers: d, a, b
    assert sorted(sel.customer_id) == ["a", "b", "d"] and len(sel) == 3
    assert select(scored, cfg, save_rate=0.05).empty


def test_learning_design_holds_out_a_random_control_from_the_pool(ds, churn):
    cfg = PolicyConfig(control_size=10)
    scored = score_candidates(ds.test, churn.predict(ds.test, ds.train), cfg)
    ld = learn_while_earning(scored, cfg, 0.3)
    t = ld["table"]
    primary = select(scored, cfg, 0.3)
    assert ld["pool_size"] == len(primary) and ld["offers"] + ld["control"] == len(primary) and ld["control"] == 10
    assert set(t.arm) == {"offer", "control"} and ld["expected_gain_forgone_vs_pure_exploitation"] > 0
    assert set(t.customer_id) == set(primary.customer_id)


def test_evaluation_scorers_and_lost_cause_penalty(ds, churn):
    p = churn.predict(ds.test, ds.train)
    rel = PolicyConfig().relevance
    lost = ds.test[ds.test.segment == "churn_intent/negative"].customer_id.head(5).tolist()
    r = reach(ds.test, lost, rel)
    assert r["churners_reached"] == 5 and r["value_reached_consistent"] == 0.0 and r["value_reached_naive"] > 0
    assert r["breakeven_save_rate"] == float("inf") and r["lost_causes_targeted"] == 5
    ev = evaluate(ds.test, p, ds.test.customer_id.head(50).tolist(), [0.3], 50, rel, random_seeds=5)
    names = [row["list"] for row in ev["rows"]]
    assert any("oracle: persuadable" in n for n in names) and any("random" in n for n in names)


def test_critic_rejects_a_bad_list(ds, churn):
    cfg = PolicyConfig()
    a = run_audit(ds)
    bad = ds.test[ds.test.segment == "churn_intent/negative"].head(3).copy()
    bad["expected_gain"] = -1.0
    ctx = {"config": cfg, "dataset": ds, "primary_list": bad, "audit": a, "churn_metrics": churn.metrics,
           "uplift": {"ipw": {"ci_low": -0.05, "ci_high": 0.05}}}
    verdict = Critic().check(ctx)
    assert not verdict["passed"]
    assert any("lost-cause" in p for p in verdict["problems"]) and any("non-positive" in p for p in verdict["problems"])


def test_pipeline_end_to_end_writes_outputs_and_passes_critic(tmp_path):
    ctx = run_pipeline(tmp_path / "run", PolicyConfig(save_rates=[0.1, 0.3]))
    assert ctx["critic"]["passed"]
    for name in ("MEMO.md", "campaign_offers.csv", "campaign_offers_with_control.csv", "audit.md", "decision_trace.jsonl",
                 "evaluation.json", "scenarios.json", "critic.json"):
        assert (tmp_path / "run" / name).exists(), name
    offers = pd.read_csv(tmp_path / "run" / "campaign_offers.csv")
    assert len(offers) * DISCOUNT <= BUDGET and set(offers.customer_id) <= set(ctx["dataset"].test.customer_id)
