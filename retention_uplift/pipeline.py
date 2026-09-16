"""
The governed pipeline: specialist stages with declared contracts, a decision trace, a critic
that checks invariants before anything is called a recommendation, and a narrator that writes
the memo from the trace. Deterministic end to end; no model API is called.

Stages
  audit      what the data can support           -> findings
  churn      calibrated P(churn | no offer)       -> probabilities + metrics
  uplift     what the discount flag shows         -> effect estimates with intervals
  policy     the 50 offers, across save rates     -> campaign lists
  evaluate   reach and NFG on the labelled test   -> comparison against baselines and the oracle
  critic     invariants and red flags             -> pass / fail with reasons
  narrate    the memo                             -> Markdown
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from . import __version__
from .audit import audit_markdown, run_audit
from .churn_model import calibration_table, fit_churn_model
from .data import Dataset, encode
from .evaluate import evaluate
from .policy import PolicyConfig, learn_while_earning, scenario_table, score_candidates, select
from .uplift import run_uplift


@dataclass
class Trace:
    path: Path
    events: List[Dict[str, Any]] = field(default_factory=list)

    def emit(self, stage: str, kind: str, **data: Any) -> None:
        ev = {"seq": len(self.events) + 1, "at": datetime.now(timezone.utc).isoformat(), "stage": stage, "kind": kind, **data}
        self.events.append(ev)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ev, default=_json_default) + "\n")


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray, pd.Series)):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    return str(o)


@dataclass
class Stage:
    name: str
    produces: List[str]
    run: Callable[[Dict[str, Any]], Dict[str, Any]]


class Critic:
    """Invariants a campaign list must satisfy before it is presented. Each check names its evidence."""

    def check(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        cfg: PolicyConfig = ctx["config"]
        test: pd.DataFrame = ctx["dataset"].test
        primary: pd.DataFrame = ctx["primary_list"]
        audit = ctx["audit"]
        problems, notes = [], []
        spend = len(primary) * cfg.discount
        if spend > cfg.budget:
            problems.append(f"budget exceeded: {spend} > {cfg.budget}")
        if primary.customer_id.duplicated().any():
            problems.append("a customer appears twice in the offer list")
        if not set(primary.customer_id) <= set(test.customer_id):
            problems.append("offer list contains ids that are not in test_new.csv")
        banned = set(audit["sure_things"]) | set(audit["lost_causes"])
        hit = primary[primary.segment.isin(banned)]
        if len(hit):
            problems.append(f"{len(hit)} offers go to sure-thing or lost-cause segments: {hit.segment.unique().tolist()}")
        if (primary.expected_gain <= 0).any():
            problems.append("an offer with non-positive expected gain was selected")
        m = ctx["churn_metrics"]
        if m.get("test_brier", 0) > 0.05:
            problems.append(f"churn model calibration is poor on test (Brier {m['test_brier']:.3f})")
        ipw = ctx["uplift"]["ipw"]
        if ipw and ipw["ci_low"] <= 0 <= ipw["ci_high"]:
            notes.append("historical data does not establish a positive discount effect; save rate is an assumption, "
                         "and the memo must say so")
        if len(primary) < cfg.max_offers:
            notes.append(f"only {len(primary)} of {cfg.max_offers} affordable offers clear the $20 bar at save rate "
                         f"{cfg.primary_save_rate}; spending the rest would lose money in expectation")
        return {"passed": not problems, "problems": problems, "notes": notes}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def run_pipeline(out_dir: Path, cfg: PolicyConfig | None = None, data_dir: Path | None = None) -> Dict[str, Any]:
    cfg = cfg or PolicyConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "decision_trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    trace = Trace(trace_path)
    ds = Dataset.load(data_dir) if data_dir else Dataset.load()
    ctx: Dict[str, Any] = {"dataset": ds, "config": cfg}
    trace.emit("pipeline", "start", version=__version__, python=platform.python_version(),
               data_sha256_12={"train_new.csv": _sha((data_dir or ds_dir()) / "train_new.csv"),
                               "test_new.csv": _sha((data_dir or ds_dir()) / "test_new.csv")},
               config={"budget": cfg.budget, "discount": cfg.discount, "save_rates": cfg.save_rates,
                       "primary_save_rate": cfg.primary_save_rate, "relevance": cfg.relevance, "control_size": cfg.control_size})

    def stage_audit(c):
        return {"audit": run_audit(c["dataset"])}

    def stage_churn(c):
        model = fit_churn_model(c["dataset"])
        p = model.predict(c["dataset"].test, c["dataset"].train)
        cal = calibration_table(p, c["dataset"].test.churn.to_numpy())
        return {"churn_model": model, "p_churn_test": p, "churn_metrics": model.metrics, "calibration": cal}

    def stage_uplift(c):
        return {"uplift": run_uplift(c["dataset"])}

    def stage_policy(c):
        scored = score_candidates(c["dataset"].test, c["p_churn_test"], cfg)
        primary = select(scored, cfg, cfg.primary_save_rate)
        return {"scored": scored, "primary_list": primary, "scenarios": scenario_table(scored, cfg),
                "learning_design": learn_while_earning(scored, cfg, cfg.primary_save_rate)}

    def stage_evaluate(c):
        return {"evaluation": evaluate(c["dataset"].test, c["p_churn_test"], c["primary_list"].customer_id.tolist(),
                                       cfg.save_rates, len(c["primary_list"]), cfg.relevance)}

    def stage_critic(c):
        return {"critic": Critic().check(c)}

    stages = [Stage("audit", ["audit"], stage_audit), Stage("churn", ["churn_metrics"], stage_churn),
              Stage("uplift", ["uplift"], stage_uplift), Stage("policy", ["primary_list", "scenarios"], stage_policy),
              Stage("evaluate", ["evaluation"], stage_evaluate), Stage("critic", ["critic"], stage_critic)]
    for st in stages:
        t0 = time.perf_counter()
        trace.emit(st.name, "started")
        result = st.run(ctx)
        ctx.update(result)
        summary = {k: _summarise(result[k]) for k in st.produces if k in result}
        trace.emit(st.name, "finished", ms=int((time.perf_counter() - t0) * 1000), **summary)

    write_outputs(out_dir, ctx, trace)
    trace.emit("pipeline", "finished", critic_passed=ctx["critic"]["passed"], offers=int(len(ctx["primary_list"])))
    return ctx


def ds_dir() -> Path:
    from .data import DATA_DIR
    return DATA_DIR


def _summarise(v: Any) -> Any:
    if isinstance(v, pd.DataFrame):
        return {"rows": int(len(v))}
    if isinstance(v, dict):
        return {k: _summarise(x) for k, x in v.items() if k not in ("segments", "table", "rows")}
    if isinstance(v, list) and len(v) > 12:
        return f"list[{len(v)}]"
    return v


def write_outputs(out_dir: Path, ctx: Dict[str, Any], trace: Trace) -> None:
    cfg: PolicyConfig = ctx["config"]
    (out_dir / "audit.md").write_text(audit_markdown(ctx["audit"]), encoding="utf-8")
    (out_dir / "audit.json").write_text(json.dumps(ctx["audit"], indent=2, default=_json_default), encoding="utf-8")
    (out_dir / "uplift.json").write_text(json.dumps(ctx["uplift"], indent=2, default=_json_default), encoding="utf-8")
    (out_dir / "churn_metrics.json").write_text(json.dumps({"metrics": ctx["churn_metrics"], "calibration": ctx["calibration"]}, indent=2, default=_json_default), encoding="utf-8")
    ctx["scored"].to_csv(out_dir / "scored_test_customers.csv", index=False)
    cols = ["rank", "customer_id", "segment", "p_churn", "estimated_clv", "value_at_risk", "breakeven_save_rate", "expected_gain"]
    ctx["primary_list"][cols].to_csv(out_dir / "campaign_offers.csv", index=False)
    ld = ctx["learning_design"]["table"]
    ld[["rank", "customer_id", "segment", "p_churn", "estimated_clv", "arm"]].to_csv(out_dir / "campaign_offers_with_control.csv", index=False)
    (out_dir / "scenarios.json").write_text(json.dumps(ctx["scenarios"], indent=2, default=_json_default), encoding="utf-8")
    (out_dir / "evaluation.json").write_text(json.dumps(ctx["evaluation"], indent=2, default=_json_default), encoding="utf-8")
    (out_dir / "critic.json").write_text(json.dumps(ctx["critic"], indent=2), encoding="utf-8")
    (out_dir / "MEMO.md").write_text(narrate(ctx), encoding="utf-8")


def narrate(ctx: Dict[str, Any]) -> str:
    cfg: PolicyConfig = ctx["config"]
    a, u, m, ev, cr = ctx["audit"], ctx["uplift"], ctx["churn_metrics"], ctx["evaluation"], ctx["critic"]
    primary = ctx["primary_list"]
    ld = ctx["learning_design"]
    segs = primary.segment.value_counts().to_dict()
    lines = ["# Retention campaign memo: 50 offers, $1,000, test_new.csv", "",
             f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by the pipeline in this repository. "
             f"Critic verdict: **{'PASS' if cr['passed'] else 'FAIL'}**" + (f" ({'; '.join(cr['problems'])})" if cr["problems"] else "") + ".", "",
             "## Recommendation", "",
             f"Send the $20 offer to the **{len(primary)}** customers in `campaign_offers.csv` (spend ${len(primary) * cfg.discount:,.0f} of the $1,000). "
             f"They are the customers with the highest expected value at risk among those a price discount can plausibly reach: "
             + ", ".join(f"{v} from {k}" for k, v in segs.items()) + ". "
             f"Nobody from the segments that never churn ({', '.join(a['sure_things'])}) or always churn regardless "
             f"({', '.join(a['lost_causes'])}) receives an offer.", "",
             f"If the campaign must also *measure* whether the discount works, use `campaign_offers_with_control.csv` instead: "
             f"{ld['offers']} offers and {ld['control']} customers held out at random from the same {ld['pool_size']}, spending "
             f"${ld['offers'] * cfg.discount:,.0f}. Expected cost of that knowledge, at save rate {cfg.primary_save_rate:.0%}: "
             f"${ld['expected_gain_forgone_vs_pure_exploitation']:,.0f} of expected gain (the offers those {ld['control']} would have received). "
             "The saved budget can go to the next-ranked customers instead, but then the control is no longer drawn from the offer pool.", "",
             "## What the historical data supports, and what it does not", ""]
    lines += [f"- {c}" for c in a["conclusions"]]
    lines += [f"- {u['conclusion']}", "",
              "## Churn model", "",
              f"Calibrated gradient boosting trained on the {m['train_rows_untreated']} untreated customers. Cross-validated AUC "
              f"{m['cv_auc_calibrated']:.3f}, Brier {m['cv_brier_calibrated']:.4f}; on the labelled test set AUC {m['test_auc']:.3f}, "
              f"Brier {m['test_brier']:.4f}, mean predicted churn {m['test_mean_predicted']:.3f} against an observed rate of {m['test_churn_rate']:.3f}. "
              "The test label was used only to score, never to fit.", "",
              "## Expected net gain by assumed save rate", "",
              "The ranking of customers does not depend on the save rate; only how many clear the $20 bar does.", "",
              "| Save rate among persuadables | Offers | Spend | Expected value saved | Expected net gain |", "|---:|---:|---:|---:|---:|"]
    for r in ctx["scenarios"]:
        lines.append(f"| {r['save_rate']:.0%} | {r['offers']} | ${r['spend']:,.0f} | ${r['expected_value_saved']:,.0f} | ${r['expected_net_gain']:,.0f} |")
    lines += ["", "## Reach on the labelled test set, against the baselines the case describes", "",
              "`churners reached` counts targeted customers who did churn without an offer (the test labels). Net Financial Gain "
              "is `s x value reached - spend`. The **consistent** scorer weights a reached churner's CLV by the same segment "
              "relevance the policy uses (0 for lost causes and sure things, 0.5 for support complaints); the **naive** scorer counts "
              "every reached churner at full value, which rewards offering discounts to customers who churn regardless. "
              "Break-even is the save rate a list needs to pay for itself under the consistent scorer.", "",
              "| List | Targeted | Churners reached | Targeted in zero-relevance segments | Value reached (consistent) | Value reached (naive) | Break-even s | "
              + " | ".join(f"NFG consistent @ s={s:g}" for s in cfg.save_rates) + " | NFG naive @ s=0.3 |",
              "|---|---:|---:|---:|---:|---:|---:|" + "---:|" * (len(cfg.save_rates) + 1)]
    for r in ev["rows"]:
        be = "∞" if r["breakeven_save_rate"] == float("inf") else f"{r['breakeven_save_rate']:.2f}"
        lines.append(f"| {r['list']} | {r['targeted']:.0f} | {r['churners_reached']:.1f} | {r['lost_causes_targeted']:.1f} | "
                     f"${r['value_reached_consistent']:,.0f} | ${r['value_reached_naive']:,.0f} | {be} | "
                     + " | ".join(f"${r[f'nfg_consistent@s={s:g}']:,.0f}" for s in cfg.save_rates)
                     + f" | ${r['nfg_naive@s=0.3']:,.0f} |")
    lines += ["", "The two scorers disagree on the risk x CLV baseline: it reaches more raw value because it targets `churn_intent` "
              "customers, who in the training data churned 15 times out of 15 *with* the discount. Whether that value is reachable "
              "is exactly the assumption the segment relevance encodes; the naive column shows what is at stake if the assumption is wrong."]
    lines += ["", "## Critic notes", ""] + [f"- {n}" for n in cr["notes"]] + [
        "", "## Assumptions that a reader should challenge", "",
        f"- Relevance multipliers by segment: {cfg.relevance}. A price discount answers a pricing complaint; the half weight on "
        "support complaints is a judgement, not a measurement.",
        "- `estimated_clv` is taken as the value lost on churn. It is monthly revenue times an estimated remaining lifetime; "
        "the discount's own $20 is not netted out of it.",
        "- The save rate is unknown. The historical flag does not establish it; the control group in the second list is how to learn it."]
    return "\n".join(lines) + "\n"
