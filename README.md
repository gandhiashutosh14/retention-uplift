# Retention Uplift: 50 offers, $1,000, maximum net gain

**Which 50 customers should get a $20 retention discount, when the data says nothing sure about whether the discount works?** A governed pipeline that audits the data, calibrates churn risk, bounds the discount's effect honestly, allocates the budget across save-rate scenarios, scores the list on the labelled test set, and refuses to present a recommendation that breaks its own rules.

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Status](https://img.shields.io/badge/status-case%20study%20prototype-orange)

## The case

A telecom's marketing team has $1,000 for next month's retention campaign on `test_new.csv`: single $20 offers, at most one per customer. Historically offers went by churn-risk score alone, which wasted money on customers who would have stayed anyway, customers who left regardless, and low-value customers. **Net Financial Gain** = value retained from customers who would have churned but were saved, minus $20 per targeted customer. Data and dictionary: [rajgupt/telco_churn_case](https://github.com/rajgupt/telco_churn_case) (cloned under `data/`).

## The answer, in one screen

`retention run --out reports/run1` produces [`reports/run1/MEMO.md`](reports/run1/MEMO.md) and [`campaign_offers.csv`](reports/run1/campaign_offers.csv). Summary of that run:

| | |
|---|---|
| Offers | 50 customers, $1,000: 31 pricing/neutral, 13 customer_support/negative, 6 pricing/negative |
| Excluded by evidence | the 379 test customers whose segment never churns, and the 48 whose segment churned 15 of 15 times *with* the discount |
| Churn model | calibrated gradient boosting fit on the 2,531 untreated rows: test AUC 0.994, Brier 0.0039, predicted churn 14.4% vs observed 14.2% (test label used for scoring only) |
| What the discount flag shows | IPW-adjusted churn reduction +0.011 with 95% bootstrap interval [−0.057, +0.078]; out-of-fold Qini −0.03 (T-learner) and +0.03 (class transformation). No demonstrated effect. |
| Reach on the test labels | 45 of the 50 targeted customers did churn without an offer; $46,158 of persuadable-weighted CLV reached, against an oracle ceiling of $48,138 |
| Expected net gain | depends on the unknown save rate *s*: $3.6K at *s* = 0.1, $12.8K at 0.3, $22.1K at 0.5; break-even *s* = 0.02 |
| Baselines the case names | churn-risk top-50 reaches $21.5K and puts 28 offers on lost causes; CLV top-50 $28.6K; risk × CLV top-50 $40.0K; random $3.7K |
| Learn-while-earning list | 40 offers + 10 held out at random from the same 50, so the campaign measures *s*; expected cost $3.5K at *s* = 0.3 |

Every number above is in the run's JSON outputs and regenerates from the command.

## What the data says, and what the pipeline does about it

1. **The discount was not assigned at random** (5% of customers; 12.7% of neutral-sentiment customers, 2% of positive). Churn is *higher* among discounted customers (20.0% vs 14.3%). A naive uplift model concludes the discount hurts; that is confounding.
2. **Two segments are decided before any offer.** `service_quality/positive` customers never churn (1,521 of 1,521). `churn_intent/negative` customers always churn, and the 15 who received the discount all churned anyway. Spending on either is waste by construction, so the policy gives them relevance 0.
3. **Where the discount was actually tried on persuadable people** (pricing complaints), churn was 15.0% without it and 13.2% with it, and every estimator's interval includes zero. The pipeline therefore does not pretend to know the save rate: it reports the decision across *s* and ships a second list with a randomised control so the campaign measures *s* itself.
4. **The ranking does not depend on *s*.** Expected gain is `s × relevance × P(churn) × CLV − 20`, and *s* multiplies every candidate alike; the knapsack with equal costs is solved exactly by sorting. Only the *number* of customers who clear the $20 bar moves with *s* (40 at 0.1, all 50 from 0.3 up).
5. **A scorer can lie by omission.** Counting every reached churner's CLV as saveable makes "risk × CLV" look best ($71K reached) because it spends 22 offers on customers who churn regardless. The memo shows both the consistent and the naive scorer so the assumption is visible rather than buried.

## The agentic part, and what it is not

The pipeline is a bounded sequence of specialist stages with declared outputs, a **decision trace** (`decision_trace.jsonl`, one event per stage with hashes of the input data and the configuration), a **critic** that checks invariants before the list is called a recommendation (budget, one offer per customer, ids exist, no sure things or lost causes, positive expected gain for every offer, calibration on the test set) and records notes the memo must carry, and a **narrator** that writes the memo from the trace. It is deterministic and calls no model API; "agent" here means a stage with a contract and a veto, not a chat loop. The critic's veto is exercised in the tests.

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]" scikit-uplift                 # scikit-uplift is used for the Qini metric and one reference learner
pytest -q                                            # 8 tests, about 30 s
retention run --out reports/run1                     # writes MEMO.md, campaign_offers.csv, the control design and all JSON
retention run --out reports/alt --primary 0.2 --support-relevance 1.0   # different assumptions, same machinery
```

## Layout

```
retention_uplift/data.py         loading, encoding, the segment key, budget constants
retention_uplift/audit.py        stage 1: segment table with Wilson bounds; sure things and lost causes from counts
retention_uplift/churn_model.py  stage 2: calibrated churn model on untreated rows; test-set scoring
retention_uplift/uplift.py       stage 3: segment difference, IPW bootstrap, T-learner, class transformation, Qini
retention_uplift/policy.py       stage 4: expected gain, exact knapsack, scenario sweep, learn-while-earning control
retention_uplift/evaluate.py     stage 5: reach on test labels, consistent vs naive NFG, baselines and oracles
retention_uplift/pipeline.py     stages, decision trace, critic, narrator
retention_uplift/cli.py          retention run
research/SOURCES.md              the repositories, datasets and papers consulted, and how they changed the design
reports/run1/                    the committed run
tests/                           audit structure, model scoring, knapsack exactness, bans, control design, critic veto, end to end
```

## Assumptions to challenge

- `estimated_clv` is treated as the value lost on churn, and no discount cost is netted from it.
- Relevance 0.5 for customer-support complaints is a judgement that a price discount half-answers a service complaint. Change it with `--support-relevance`.
- The save rate is unknown. Nothing in the historical data pins it down; the control group is how to learn it.
- The dataset is synthetic and near-deterministic (churn AUC 0.99). Real customers will not separate this cleanly.

## License

MIT for the code. The dataset belongs to its repository's author.
