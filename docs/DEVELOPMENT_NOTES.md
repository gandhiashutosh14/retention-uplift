# Development notes

Built from scratch on 2026-09-17 from the case-study brief (`round2_case_study.docx`) and the
dataset repository it names. No employer code or data was used.

## Order of work

1. Read the brief; cloned `rajgupt/telco_churn_case`; profiled train and test.
2. Research on public uplift datasets, comparable repositories and budget-constrained uplift
   papers (`research/SOURCES.md`). The one lesson that changed the design: every serious source
   has a randomised treatment, and this dataset does not.
3. Data audit before modelling. It found the three facts the whole solution rests on: discount
   assignment depends on sentiment, two segments have deterministic churn, and the only persuadable
   segment with treated rows shows no measurable effect.
4. Pipeline: audit, calibrated churn model on untreated rows, three uplift estimators with
   intervals, exact knapsack across save-rate scenarios, a learn-while-earning control design,
   test-set evaluation against the baselines the case names, a critic, a narrator.

## What the tests and the first run caught

- **The first evaluation scorer rewarded targeting lost causes.** The "risk × CLV" baseline
  looked best because it counted the CLV of `churn_intent` customers as saveable, though 15 of 15
  such customers churned with the discount in training. The scorer now weights reached value by the
  same segment relevance as the policy, and the memo prints the naive number beside it.
- **A hand-written Qini normalisation was wrong** (it printed −10.5). Replaced by scikit-uplift's
  `qini_auc_score`; the two learners now score −0.03 and +0.03, which is the honest answer.
- **The learn-while-earning design drew its pool from the top 60**, but only 50 customers clear the
  gain bar at the primary save rate, so a test failed. The control is now held out from the 50
  the pure policy would target, which is also the right experiment.
- **A sign error** printed the control design's cost as a negative number.

## Verification

| Check | Result |
|---|---|
| `pytest -q` | 8 passed |
| `retention run --out reports/run1` | critic PASS; 50 offers; outputs committed under `reports/run1/` |

## What is and is not claimed

The churn model's numbers are on this synthetic, near-deterministic dataset. The uplift intervals
are what the historical flag supports and nothing more. The expected net gains are conditional on
an assumed save rate that the data does not establish; the memo says so in every section that uses
one. The "agentic" pipeline is deterministic stages with contracts and a critic, not a language
model, and the repository does not claim otherwise.
