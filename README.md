# Donor Propensity Model

Predicting donor response and retention likelihood from historical giving behavior, using real direct-mail fundraising data. Built to explore how data science can drive donor engagement and retention strategy for a fundraising/advancement organization.

## Problem Statement

Nonprofit and advancement organizations rely on mailed or targeted outreach to solicit donations, but outreach has a real cost — postage, printing, staff time — whether or not a donor responds. The core business question:

> **Which donors are most likely to respond to a solicitation, and how should limited outreach resources be prioritized to maximize response and retention rather than just contact volume?**

This project builds a propensity model to answer that question, using the same kind of RFM (recency, frequency, monetary) and demographic signals a real advancement analytics team would have access to.

## Data

- **Source:** [KDD Cup 1998](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html) — a real dataset donated by a U.S. veterans' charity, containing ~95,000 donor records with giving history, demographics, and response outcomes to a past mailing campaign.
- **Target variable:** `TARGET_B` (binary — did the donor respond to the mailing).
- **Why this dataset:** it's real fundraising response data (not a synthetic or generic churn dataset), so the features and problem framing map closely onto donor propensity/retention work in an advancement analytics context.

## Approach

1. **Exploratory Data Analysis** — assessed distributions, missingness, and class balance across ~480 raw fields; identified a relevant subset of features given the size and quality of the raw data.
2. **Feature Engineering** — built RFM-style features (recency of last gift, frequency of giving, average/lifetime gift amount) plus selected demographic fields, informed by the data dictionary.
3. **Modeling** — trained and compared baseline (logistic regression) and ensemble (random forest / XGBoost) classifiers; evaluated using precision, recall, and ROC-AUC rather than raw accuracy, given the class imbalance typical of direct-mail response data.
4. **Deployment** — serialized the trained model and served it via a FastAPI endpoint (`/predict`) that returns a donor's propensity score given their feature values; also deployed as a managed endpoint via Azure ML.

## Results

*(To be filled in once modeling is complete)*

| Model | Precision | Recall | ROC-AUC |
|---|---|---|---|
| Logistic Regression | — | — | — |
| Random Forest | — | — | — |

**Key findings:** *(e.g., which features were most predictive of response — to be added)*

## Repo Structure

```
donor-propensity-model/
├── data/               # raw and processed data (not committed - see .gitignore)
├── notebooks/          # EDA, feature engineering, and modeling notebooks
├── src/                # reusable data processing / feature engineering code
├── api/
│   └── main.py         # FastAPI serving endpoint
├── model.pkl           # trained model artifact (not committed if large - see notes)
├── requirements.txt
└── README.md
```

## Running Locally

```bash
# install dependencies
pip install -r requirements.txt

# run the API (requires model.pkl to exist - see notebooks/ to train one)
uvicorn api.main:app --reload
```

Then test the endpoint:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"recency_months": 6, "frequency": 4, "avg_gift_amount": 25.50, "age": 55, "years_since_first_gift": 8}'
```

Interactive API docs available at `http://127.0.0.1:8000/docs` once running.

## Extending This for a Real Advancement Organization

In a production setting, this model would ideally:
- Integrate directly with CRM platforms (e.g., Salesforce/CRM Analytics) so propensity scores surface in the tools gift officers already use.
- Retrain on a regular cadence as new giving/response data comes in.
- Incorporate cost-per-contact into scoring, so recommendations optimize net expected return rather than raw response probability alone.

## Tech Stack

Python, pandas, scikit-learn, FastAPI, Azure ML

---
*Built as a self-directed learning project to apply predictive modeling and deployment skills to a fundraising/advancement analytics context.*