# Data audit

Train 2666 rows, test 667 rows. Test carries the churn label (yes) and the discount column (no). Churn rate train 14.6%, test 14.2%. Discount rate train 5.1% (135 customers, of whom 27 churned).

| Segment (feedback category / sentiment) | n | treated | churn, no discount | churn, discount | naive uplift | 95% bounds on uplift | mean CLV |
|---|---:|---:|---:|---:|---:|---|---:|
| churn_intent/negative | 218 | 15 | 100.0% | 100.0% | +0.000 | [-0.019, +0.204] | 874 |
| customer_support/negative | 123 | 0 | 41.5% |  |  |  | 717 |
| pricing/negative | 86 | 0 | 33.7% |  |  |  | 734 |
| pricing/neutral | 718 | 91 | 12.4% | 13.2% | -0.007 | [-0.116, +0.075] | 1,182 |
| service_quality/positive | 1521 | 29 | 0.0% | 0.0% | +0.000 | [-0.117, +0.003] | 552 |

## Conclusions

- The discount was given to 135 of 2666 customers (5.1%), not at random: the rate varies by sentiment {'negative': 0.035, 'neutral': 0.127, 'positive': 0.019}.
- Churn among discounted customers (20.0%) is higher than among the rest (14.3%); a naive comparison says the discount hurts, which is confounding, not causation.
- Segments with deterministic churn in the data: ['churn_intent/negative', 'service_quality/positive']. Sure things (never churn): ['service_quality/positive']. Lost causes (always churn, discount or not): ['churn_intent/negative']. Spending on either is wasted by definition.
- Within the only segment where the discount was tried on customers who both could and did not always churn (pricing complaints), the Wilson-interval bounds on its effect include zero: the historical data cannot show that the $20 offer saves anyone. The targeting policy therefore treats the save rate as an assumption to be learned from the campaign itself, and reports the decision across a range of save rates.
- feedback_text has 5 distinct values, fully captured by feedback_category and sentiment; no text model is needed.
