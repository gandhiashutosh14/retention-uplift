# Retention campaign memo: 50 offers, $1,000, test_new.csv

Generated 2026-09-16 23:35 UTC by the pipeline in this repository. Critic verdict: **PASS**.

## Recommendation

Send the $20 offer to the **50** customers in `campaign_offers.csv` (spend $1,000 of the $1,000). They are the customers with the highest expected value at risk among those a price discount can plausibly reach: 31 from pricing/neutral, 13 from customer_support/negative, 6 from pricing/negative. Nobody from the segments that never churn (service_quality/positive) or always churn regardless (churn_intent/negative) receives an offer.

If the campaign must also *measure* whether the discount works, use `campaign_offers_with_control.csv` instead: 40 offers and 10 customers held out at random from the same 50, spending $800. Expected cost of that knowledge, at save rate 30%: $3,473 of expected gain (the offers those 10 would have received). The saved budget can go to the next-ranked customers instead, but then the control is no longer drawn from the offer pool.

## What the historical data supports, and what it does not

- The discount was given to 135 of 2666 customers (5.1%), not at random: the rate varies by sentiment {'negative': 0.035, 'neutral': 0.127, 'positive': 0.019}.
- Churn among discounted customers (20.0%) is higher than among the rest (14.3%); a naive comparison says the discount hurts, which is confounding, not causation.
- Segments with deterministic churn in the data: ['churn_intent/negative', 'service_quality/positive']. Sure things (never churn): ['service_quality/positive']. Lost causes (always churn, discount or not): ['churn_intent/negative']. Spending on either is wasted by definition.
- Within the only segment where the discount was tried on customers who both could and did not always churn (pricing complaints), the Wilson-interval bounds on its effect include zero: the historical data cannot show that the $20 offer saves anyone. The targeting policy therefore treats the save rate as an assumption to be learned from the campaign itself, and reports the decision across a range of save rates.
- feedback_text has 5 distinct values, fully captured by feedback_category and sentiment; no text model is needed.
- Within the 718 customers in segments where the discount was tried on people who could go either way (pricing/neutral), the IPW-adjusted churn reduction is +0.011 with a 95% bootstrap interval [-0.057, +0.078]. The interval includes zero: the data does not demonstrate that the discount saves anyone. Out-of-fold normalised Qini coefficients: T-learner -0.030, class transformation +0.030 (zero is random ordering, one is perfect). The policy therefore treats the save rate as a scenario parameter and the campaign as the experiment that measures it.

## Churn model

Calibrated gradient boosting trained on the 2531 untreated customers. Cross-validated AUC 0.987, Brier 0.0065; on the labelled test set AUC 0.994, Brier 0.0039, mean predicted churn 0.144 against an observed rate of 0.142. The test label was used only to score, never to fit.

## Expected net gain by assumed save rate

The ranking of customers does not depend on the save rate; only how many clear the $20 bar does.

| Save rate among persuadables | Offers | Spend | Expected value saved | Expected net gain |
|---:|---:|---:|---:|---:|
| 10% | 40 | $800 | $4,598 | $3,798 |
| 20% | 49 | $980 | $9,467 | $8,487 |
| 30% | 50 | $1,000 | $14,223 | $13,223 |
| 50% | 50 | $1,000 | $23,705 | $22,705 |
| 100% | 50 | $1,000 | $47,411 | $46,411 |

## Reach on the labelled test set, against the baselines the case describes

`churners reached` counts targeted customers who did churn without an offer (the test labels). Net Financial Gain is `s x value reached - spend`. The **consistent** scorer weights a reached churner's CLV by the same segment relevance the policy uses (0 for lost causes and sure things, 0.5 for support complaints); the **naive** scorer counts every reached churner at full value, which rewards offering discounts to customers who churn regardless. Break-even is the save rate a list needs to pay for itself under the consistent scorer.

| List | Targeted | Churners reached | Targeted in zero-relevance segments | Value reached (consistent) | Value reached (naive) | Break-even s | NFG consistent @ s=0.1 | NFG consistent @ s=0.2 | NFG consistent @ s=0.3 | NFG consistent @ s=0.5 | NFG consistent @ s=1 | NFG naive @ s=0.3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| policy (segment-aware expected gain) | 50 | 45.0 | 0.0 | $46,158 | $49,053 | 0.02 | $3,616 | $8,232 | $12,847 | $22,079 | $45,158 | $13,716 |
| churn-risk top-n | 50 | 50.0 | 28.0 | $21,533 | $46,581 | 0.05 | $1,153 | $3,307 | $5,460 | $9,766 | $20,533 | $12,974 |
| clv top-n | 50 | 35.0 | 15.0 | $28,615 | $51,026 | 0.03 | $1,862 | $4,723 | $7,585 | $13,308 | $27,615 | $14,308 |
| risk x clv top-n (no segment logic) | 50 | 50.0 | 22.0 | $39,956 | $71,113 | 0.03 | $2,996 | $6,991 | $10,987 | $18,978 | $38,956 | $20,334 |
| oracle: persuadable churners with highest weighted clv | 50 | 50.0 | 3.0 | $48,138 | $52,768 | 0.02 | $3,814 | $8,628 | $13,441 | $23,069 | $47,138 | $14,830 |
| oracle (naive): any churners with highest clv | 50 | 50.0 | 22.0 | $39,378 | $71,234 | 0.03 | $2,938 | $6,876 | $10,813 | $18,689 | $38,378 | $20,370 |
| random n (mean of 200 draws) | 50 | 7.3 | 31.9 | $3,743 | $7,448 | ∞ | $-626 | $-251 | $123 | $871 | $2,743 | $1,235 |

The two scorers disagree on the risk x CLV baseline: it reaches more raw value because it targets `churn_intent` customers, who in the training data churned 15 times out of 15 *with* the discount. Whether that value is reachable is exactly the assumption the segment relevance encodes; the naive column shows what is at stake if the assumption is wrong.

## Critic notes

- historical data does not establish a positive discount effect; save rate is an assumption, and the memo must say so

## Assumptions that a reader should challenge

- Relevance multipliers by segment: {'pricing/neutral': 1.0, 'pricing/negative': 1.0, 'customer_support/negative': 0.5, 'service_quality/positive': 0.0, 'churn_intent/negative': 0.0}. A price discount answers a pricing complaint; the half weight on support complaints is a judgement, not a measurement.
- `estimated_clv` is taken as the value lost on churn. It is monthly revenue times an estimated remaining lifetime; the discount's own $20 is not netted out of it.
- The save rate is unknown. The historical flag does not establish it; the control group in the second list is how to learn it.
