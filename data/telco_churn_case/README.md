This repo contains dataset for telco case study

## Data Dictionary
| Column                       | Type                 | Description                                                             |
| ---------------------------- | -------------------- | ----------------------------------------------------------------------- |
| customer\_id                 | String               | Unique customer identifier.                                             |
| state                        | Categorical          | Customer's state of residence.                                          |
| account\_length              | Integer              | Number of days/months the customer account has been active.             |
| area\_code                   | Categorical          | Telecom area code (e.g., 408, 415, 510).                                |
| international\_plan          | Categorical (Yes/No) | Indicates whether customer subscribes to an international calling plan. |
| voice\_mail\_plan            | Categorical (Yes/No) | Indicates whether customer subscribes to voicemail service.             |
| number\_vmail\_messages      | Integer              | Number of voicemail messages.                                           |
| total\_day\_minutes          | Numeric              | Total daytime usage minutes.                                            |
| total\_day\_calls            | Integer              | Number of daytime calls.                                                |
| total\_day\_charge           | Numeric              | Charges incurred during daytime usage.                                  |
| total\_eve\_minutes          | Numeric              | Total evening usage minutes.                                            |
| total\_eve\_calls            | Integer              | Number of evening calls.                                                |
| total\_eve\_charge           | Numeric              | Charges incurred during evening usage.                                  |
| total\_night\_minutes        | Numeric              | Total nighttime usage minutes.                                          |
| total\_night\_calls          | Integer              | Number of nighttime calls.                                              |
| total\_night\_charge         | Numeric              | Charges incurred during nighttime usage.                                |
| total\_intl\_minutes         | Numeric              | Total international calling minutes.                                    |
| total\_intl\_calls           | Integer              | Number of international calls.                                          |
| total\_intl\_charge          | Numeric              | Charges incurred from international calls.                              |
| customer\_service\_calls     | Integer              | Number of calls made to customer service.                               |
| churn                        | Binary (0/1)         | Target variable indicating whether the customer churned.                |
| split                        | Categorical          | Dataset partition indicator (e.g., train/test).                         |
| has\_international\_plan     | Binary (0/1)         | Encoded version of international\_plan.                                 |
| has\_voice\_mail\_plan       | Binary (0/1)         | Encoded version of voice\_mail\_plan.                                   |
| total\_minutes               | Numeric              | Sum of day, evening, night, and international minutes.                  |
| total\_calls                 | Integer              | Sum of day, evening, night, and international calls.                    |
| total\_charges               | Numeric              | Sum of all charge components.                                           |
| monthly\_revenue             | Numeric              | Estimated monthly customer revenue.                                     |
| estimated\_remaining\_months | Integer              | Predicted remaining customer lifetime in months.                        |
| estimated\_clv               | Numeric              | Estimated Customer Lifetime Value (CLV).                                |
| feedback\_text               | Text                 | Customer feedback or comment.                                           |
| feedback\_category           | Categorical          | Categorized feedback reason.                                            |
| sentiment                    | Categorical          | Sentiment extracted from feedback text.                                 |
| complaint\_intensity         | Integer (1–5)        | Severity score of complaint or dissatisfaction.                         |
| discount                     | Binary (0/1)         | $20 discount given (yes/no)                                             |

