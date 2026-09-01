# Donor Propensity Model

Predicting donor response and retention likelihood from historical giving behavior, using real direct-mail fundraising data. Built to explore how data science can drive donor engagement and retention strategy for a fundraising/advancement organization.

## Problem Statement

Nonprofit and advancement organizations rely on mailed or targeted outreach to solicit donations, but outreach has a real cost — postage, printing, staff time — whether or not a donor responds. The core business question:

> **Which donors are most likely to respond to a solicitation, and how should limited outreach resources be prioritized to maximize response and retention rather than just contact volume?**

This project builds a propensity model to answer that question, using the same kind of RFM (recency, frequency, monetary) and demographic signals a real advancement analytics team would have access to.

## Data

- **Source:** [KDD Cup 1998](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html) — a real dataset donated by a U.S. veterans' charity (PVA), containing donor records with giving history, demographics, and response outcomes to a June 1997 renewal mailing targeted at lapsed donors.
- **Population:** ~95,412 lapsed PVA donors. Response rate to the mailing was **5.1%** — a realistic, heavily imbalanced direct-mail response rate.
- **Target variable:** `TARGET_B` (yes/no — did the donor respond to the mailing?).

## Approach

### 1. Exploratory Data Analysis
- Loaded the raw ~481-column dataset and identified uninformative columns based on missingness (e.g., 20 columns tied to donor history on unrelated past campaigns had >90% missing data) — these columns were dropped.
- Confirmed the ~5.1% response rate and its implications for modeling (accuracy is a misleading metric on this data, as a model that always predicts "no response" would already be ~95% "accurate").

### 2. Feature Engineering
Selected and engineered 12 features grounded in RFM logic, cross-referenced against the dataset's data dictionary:

| Feature | Type | Notes |
|---|---|---|
| `NGIFTALL`, `CARDGIFT` | Frequency | Lifetime gift counts (all gifts vs. card-promotion gifts specifically) |
| `AVGGIFT`, `MINRAMNT`, `MAXRAMNT` | Monetary | Average / smallest / largest gift amount to date |
| `TIMELAG` | Behavioral | Months between first and second gift; missing values (single-gift donors) were filled with 0, since the gap is structurally undefined rather than unknown |
| `MONTHS_SINCE_LAST_GIFT`, `MONTHS_SINCE_FIRST_GIFT` | Recency / tenure | Derived from raw `YYMM`-encoded date fields, calculated relative to the actual June 1997 mailing date |
| `AGE`, `INCOME` | Demographic | Household income is an ordinal 1–7 bracket, not a dollar figure (assumed to indicate an income range e.g. 1=Income < $15,000, 7=Income >= $150,000) |
| `AGE_MISSING`, `INCOME_MISSING` | Missingness flags | ~25% and ~22% of donors respectively lacked `AGE` and `INCOME` values; they were assigned the median and a "was this assigned" flag rather than dropping the fields, to preserve the information that a value was originally unknown |

One field (`RFA_2R`, a pre-built recency code) was found to be constant across the entire population — consistent with the dataset's lapsed-donor-only sampling design — and was dropped as uninformative.

### 3. Modeling
Trained and compared three models, using `class_weight='balanced'` (or the XGBoost equivalent, `scale_pos_weight`) to address the class imbalance:

| Model | ROC-AUC | Precision (responders) | Recall (responders) |
|---|---|---|---|
| **Logistic Regression** | **0.604** | 0.07 | 0.56 |
| Random Forest | 0.567 | 0.03 | 0.01 |
| XGBoost | 0.559 | 0.07 | 0.36 |

**Logistic regression (the simplest model)  performed best**, and consistently so across all three trials. Feature-importance analysis showed that predictive signal was spread fairly evenly across all 12 features rather than concentrated in a few strong predictors, which favors a simpler model less prone to overfitting on a modest, mostly-linear feature set.

### 4. Threshold Analysis
Swept the classification threshold from 0.5 to 0.8 and found precision improved only marginally (0.07 → 0.20) while recall collapsed almost entirely (0.56 → 0.00). Given the real-world cost structure (a mailing costs cents, while a missed donor is a fully lost donation) **a lower threshold (0.5 or below) is the better real-world choice**, prioritizing catching more true responders over avoiding false positives.

### 5. Deployment
- **Local:** served via a FastAPI `/predict` endpoint, returning a propensity score and classification for a given donor's feature values.
- **Cloud:** registered the trained model to Azure ML and deployed a live, verified real-time endpoint (see below).

## Azure ML Deployment

Registered the model to Azure ML Studio and deployed it as a managed real-time endpoint (`Standard_DS1_v2`, cost-optimized single instance), with a custom scoring script (`azure_scoring_script.py`) and a custom conda environment matching the exact local package versions (numpy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0, Python 3.12).

Getting to a working deployment involved diagnosing and resolving several infrastructure issues along the way:
- A subscription resource-provider registration gap (`Microsoft.PolicyInsights`) blocking endpoint provisioning.
- A pickle/numpy version mismatch between the local training environment and Azure's curated `sklearn-1.5` image (`numpy._core` didn't exist in the older numpy Azure shipped).
- A custom Docker-context environment that silently didn't apply its conda dependencies, traced to a missing `COPY`/`RUN conda env update` step.
- A legacy `azureml-defaults` dependency pulling in an incompatible old `pyarrow` build.

The final deployment was verified end-to-end: a live request to the Azure endpoint returned a propensity score of **0.5132** for a test donor — identical to the score returned by the local FastAPI endpoint for the same input, confirming the deployed model matches the locally validated one.

## Repo Structure

```
donor-propensity-model/
├── data/               # to store raw and processed data (not committed - see .gitignore)
├── notebooks/          # EDA, feature engineering, and modeling via Jupyter Notebook
├── api/
│   ├── main.py                 # FastAPI serving endpoint
│   ├── model.pkl                # trained model artifact
│   └── azure_scoring_script.py # Azure ML scoring script (init/run)
├── conda_env.yml       # Azure ML custom environment definition
├── requirements.txt
└── README.md
```

## Running Locally

```bash
# install dependencies
pip install -r requirements.txt
cd api
# run the API (requires model.pkl to exist - see notebooks/ to train one)
uvicorn main:app --reload
```

Then test the endpoint (should output a propensity_score and predicted_class):
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"NGIFTALL": 12, "CARDGIFT": 8, "AVGGIFT": 15.50, "MINRAMNT": 5.0, "MAXRAMNT": 25.0, "TIMELAG": 3.0, "MONTHS_SINCE_LAST_GIFT": 18, "MONTHS_SINCE_FIRST_GIFT": 91, "AGE": 62, "AGE_MISSING": 0, "INCOME": 4, "INCOME_MISSING": 0}'
```

Interactive API docs available at `http://127.0.0.1:8000/docs`.

## Key Takeaways

- **Simpler isn't always worse** — logistic regression outperformed both random forest and XGBoost on this dataset, a genuine, evidence-backed finding rather than a default assumption.
- **Threshold choice is a business decision, not just a statistical one** — the "right" cutoff depends on the real relative cost of a wasted contact vs. a missed donor.
- **Missing data needs a reason, not just a fill** — `TIMELAG`'s missingness was structural (no second gift), while `AGE`/`INCOME`'s was likely non-disclosure; each was handled differently as a result.
- **Deployment is rarely the smooth part** — getting a model from a notebook to a live, cloud-served endpoint surfaced several real environment and dependency mismatches, each requiring genuine debugging to resolve.

## Extending This for a Real Advancement Organization

In a production setting, this model would ideally:
- Integrate directly with CRM platforms (e.g., Salesforce/CRM Analytics) so propensity scores surface in the tools gift officers already use.
- Retrain on a regular cadence as new giving/response data comes in.
- Incorporate cost-per-contact into scoring, so recommendations optimize net expected return rather than raw response probability alone.

## Tech Stack

Python, pandas, scikit-learn, FastAPI, Azure ML

---
*Built as a self-directed learning project applying predictive modeling and deployment skills to a fundraising/advancement analytics context.*