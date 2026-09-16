# Research: repositories, datasets and methods for this problem

Compiled 2026-09-17. Everything below was fetched and read; nothing is quoted from memory.

## The assigned dataset

- **rajgupt/telco_churn_case** (GitHub): `train_new.csv` (2,666 rows, 35 columns) and `test_new.csv`
  (667 rows, 34 columns; no `discount` column, but it does carry `churn`), plus the data dictionary
  in its README. Cloned into `data/telco_churn_case/` for this project. The columns are the classic
  BigML/Kaggle "telecom churn" schema (state, plans, minutes/calls/charges by period, service calls)
  extended with synthetic CLV (`monthly_revenue x estimated_remaining_months`), templated feedback
  text with five distinct sentences, a feedback category, sentiment, complaint intensity, and a
  historical `$20 discount` flag.

## Public datasets with a real treatment flag (for method validation, not used here)

| Dataset | Treatment | Size | Where | Why it matters here |
|---|---|---|---|---|
| Orange Belgium churn uplift dataset (arXiv 2312.07206, ECML-PKDD 2023 uplift workshop) | retention offer, randomised | telecom churn, first public churn-uplift benchmark | arxiv.org/abs/2312.07206 | the only public telecom churn dataset with a *randomised* retention treatment; the natural place to validate an uplift learner before trusting it on observational data like ours |
| Criteo Uplift | advertising exposure, randomised incrementality tests | millions of rows | `sklift.datasets.fetch_criteo` | large randomised treatment; standard Qini/AUUC benchmark |
| Hillstrom MineThatData | e-mail campaign, randomised thirds | 64,000 customers | `sklift.datasets.fetch_hillstrom` | small, classic; two treatments and a control |
| Lenta | marketing communication | 687,029 rows | `sklift.datasets.fetch_lenta` | retail, binary response |
| X5 RetailHero | communication flag | retail purchases | `sklift.datasets.fetch_x5` | treatment flag plus purchase target |

Source: scikit-uplift dataset docs at uplift-modeling.com (v0.5.1).

## Repositories that attack the same business question

- **eurookim/telco-customer-retention-analytics**: IBM Telco (7,043 customers); a randomised A/B
  test analysed with intent-to-treat and complier average causal effect, and a gradient-boosting
  churn model with a cost-based threshold under an outreach budget (AUC 0.842; claims about $136K
  over random targeting). Closest in spirit; its treatment effect comes from an actual experiment.
- **arturodeleon19/Telco_Customer_Churn**: churn model plus an evaluation function that nets CLV
  against incentive cost ($140.5K gross saved vs $14K spend in its projection).
- **manaer6-alt/02_telco_customer_churn**: threshold selection by relative costs; explicitly
  recommends uplift modelling for a production study.

## Methods

- **Two-stage uplift + knapsack** is the dominant pattern: estimate CATE, then allocate under a
  budget as a (multiple-choice) knapsack. Sources: "End-to-End Cost-Effective Incentive
  Recommendation under Budget Constraint with Uplift Modeling" (RecSys 2024, arXiv 2408.11623);
  "E-Commerce Promotions Personalization via Online Multiple-Choice Knapsack with Uplift Modeling"
  (arXiv 2108.13298); "Budget-Constrained Causal Bandits" (arXiv 2604.26169) for the sequential
  version.
- **Guardrailed Uplift Targeting** (arXiv 2512.19805) formalises `max Σ π_ik w_i τ̂_k(x_i)` subject
  to budget, revenue-protection and fairness constraints, with causal forests, double machine
  learning and doubly-robust forests as learners and cumulative uplift / IPS as offline metrics.
  Notably it carries no uncertainty quantification, which is the gap this project fills with
  interval bounds and save-rate scenarios.
- **Library**: scikit-uplift 0.5.1 (`ClassTransformation`, `TwoModels`, `SoloModel`, Qini/AUUC
  metrics) is used here for the class-transformation reference and the Qini coefficient.

## What the research changed in the design

1. Every serious source uses a *randomised* treatment. Ours is observational, 5% treated, and
   assigned by sentiment. So the pipeline reports interval bounds (Wilson, IPW bootstrap) instead of
   a point uplift, and makes the campaign itself the experiment (a held-out control from the offer
   pool).
2. The knapsack with a single offer size and equal costs is solved exactly by sorting; the
   multiple-choice machinery in the papers is unnecessary here and would hide the simple fact that
   the ranking does not depend on the unknown save rate.
3. The segment logic (sure things, lost causes, persuadables) comes straight from the data audit,
   which is where the uplift literature's "four quadrants" framing earns its keep on this dataset.
