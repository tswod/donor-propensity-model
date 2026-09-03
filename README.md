# Donor Propensity Model

Predicting donor response and retention likelihood from historical giving behavior, using real direct-mail fundraising data. Built to explore how data science can drive donor engagement and retention strategy for a fundraising/advancement organization.

## Table of Contents
* [Background](#background)
* [Data](#data)
* [Approach](#approach)
* [Repo Structure](#repo-structure)
* [Running Locally](#running-locally)
* [Azure ML Deployment](#azure-ml-deployment)
* [Personal Takeaways](#personal-takeaways)
* [Tech Stack](#tech-stack)

## Background

This project explores different data modeling techniques in an attempt to replicate as closely as possible the actual results of the [KDD Cup 1998](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html) dataset, a real dataset donated by a U.S. veterans' charity (PVA) containing donor records with giving history, demographics, and response outcomes to a June 1997 mailing. 

The dataset includes a `TARGET_B` variable, which is the actual, historical outcome of the mailing outreach, indicating (YES/NO) whether a donor really responded to the 1997 mailing which targeted more than 95,000 past donors.  Of those 95,000, less than 5,000 responded — a response rate of ~5%.

Although in this case the cost of donor outreach (postage, printing, staff time, etc.) may have been relatively cheap, there could be situations where a non-response could be more costly, and a nonprofit or other advancement organization could not afford to run an outreach campaign with such a low response rate. This is where propensity modeling using RFM (recency, frequency, monetary) and demographic signals could optimize our approach to provide more insight into which donors to target BEFORE outreach, narrowing the target population to lower the initial cost while maintaining a similar overall number of responders.

The goal of this project is to attempt to answer this core business question, WITHOUT using the `TARGET_B` variable: 

> **Which donors are most likely to respond to a solicitation, and how should limited outreach resources be prioritized to maximize response and retention?**

After creating a propensity model, we can examine its effectiveness by comparing the model results to the actual, historical `TARGET_B` results. 

Once we identified our most effective model, we created a live, testable API endpoint in Azure Machine Learning Studio (has since been taken down to avoid unnecessary cloud costs) that could be used to input the demographics of a potential donor (see 'Feature Engineering' in the Approach section) and output a score between 0 and 1, where a score >0.5 would indicate the donor is more likely to respond than not (`TARGET_B='YES'`). 

In a production setting for a real organization, this could eventually be extended to:
- Create API endpoints that allow many potential donors to be scored at once rather than one at a time.
- Integrate directly with CRM platforms (e.g., Salesforce/CRM Analytics) so propensity scores become automatically included in the tools already in use.
- Retrain regularly as new giving/response data comes in in order to improve the model's effectiveness.
- Incorporate cost-per-contact into scoring, so recommendations optimize for net expected return rather than raw response probability alone.

## Data

- **Source:** [KDD Cup 1998](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html) — a real dataset donated by a U.S. veterans' charity (PVA), containing donor records with giving history, demographics, and response outcomes to a June 1997 renewal mailing targeted at lapsed donors.
- **Population:** ~95,412 lapsed PVA donors. Response rate to the mailing was **5.1%** — a realistic, heavily imbalanced direct-mail response rate.
- **Target variable:** `TARGET_B` (yes/no — did the donor respond to the mailing?).

## Approach

### Summary
We compared three modeling approaches (logistic regression, random forest, XGBoost) and selected the best-performing one by testing each model against a separate group of donors that it had never seen during training."

### 1. Exploratory Data Analysis
- Loaded the raw ~481-column dataset and identified uninformative columns based on missingness (e.g., 20 columns tied to donor history on unrelated past campaigns had >90% missing data) — these columns were dropped.
- Confirmed the ~5.1% response rate and its implications for modeling (accuracy is a misleading metric on this data, as a model that always predicts "no response" would already be ~95% "accurate").

### 2. Feature Engineering
Selected 12 features grounded in RFM logic from the dataset's data dictionary:

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
Trained and compared three models — logistic regression, random forest, and XGBoost — using `class_weight='balanced'` (or the XGBoost equivalent, `scale_pos_weight`) to address the class imbalance and emphasize minimizing mistakes on Responders compared to Non-Responders. Each model was evaluated on donors it hadn't seen during training, to get an honest read on real-world performance.

### 4. Results 
**Logistic regression (the simplest model)  performed best**, and consistently so across all three trials:

| Model | ROC-AUC | Precision (responders) | Recall (responders) |
|---|---|---|---|
| **Logistic Regression** | **0.604** | 0.07 | 0.56 |
| Random Forest | 0.567 | 0.03 | 0.01 |
| XGBoost | 0.559 | 0.07 | 0.36 |

- **ROC-AUC**: A single overall "how good is this model at telling responders and non-responders apart" score, on a scale where 0.5 means "no better than random guessing" and 1.0 means "perfect." 0.604 means the model has real, modest predictive power. Better than chance, but far from a strong signal. This is not unusual for predicting individual human behavior from historical data, especially the KDD Cup 98 dataset which is notoriously difficult to model; people are hard to predict.
- **Precision**: "When the model says 'yes, this person will likely respond,' how often is it actually right?" A precision of 0.07 means: only 7 out of every 100 people the model flags will actually respond. That sounds low, but is still somewhat better than random guessing and also exceeds the actual results where only about 5 out of 100 random people responded.
- **Recall**: "Of everyone who really would respond, how many did the model successfully catch?" A recall of 0.56 means: the model catches about 56% of the real responders — a little better than a coin flip, meaning it's missing close to half of the true opportunities, but capturing more than half.

Feature-importance analysis showed that predictive signal was spread fairly evenly across all 12 features (none of our 12 selected features stood out as especially significant/more predictive than the others), which favors a simpler model less prone to overfitting on a modest, mostly-linear feature set.

### 5. Threshold Analysis
The model gives a probability (0 to 1) that a donor will respond. We have to pick a cutoff: above what probability do we call someone "likely to respond" and mail them?

By default, that cutoff is 0.5 (50%). We tested what happens if we raise it, requiring the model to be more confident before flagging someone:

| Threshold | Precision | Recall | Real responders caught (out of 969) |
|---|---|---|---|
| 0.5 | 0.07 | 0.56 | ~544 |
| 0.6 | 0.10 | 0.15 | ~145 |
| 0.7 | 0.12 | 0.02 | ~19 |
| 0.8 | 0.20 | 0.00 | ~0 |

Raising the cutoff barely improves precision (how often the model is right when it says "yes") but causes recall (how many real responders get caught) to collapse almost completely. In other words: being pickier about who to mail doesn't meaningfully reduce wasted mailings, but it does cause us to miss the vast majority of donors who would have actually given.

Since a mailing costs relatively little (postage, printing) compared to the value of a donation, missing a real donor is far more costly than wasting a mailing on someone who doesn't respond. That means a lower cutoff — mailing more people, even at a higher false-alarm rate — is the better real-world choice, since it catches more actual donors without a large offsetting cost.

*(Follow-up: since low mailing cost favors casting an even wider net, we also tested lowering the threshold below 0.5 — see results below.)*

### 6. Deployment
- **Local:** served via a FastAPI `/predict` endpoint, returning a propensity score and classification for a given donor's feature values.
- **Cloud:** registered the trained model to Azure ML and deployed a live, verified real-time endpoint (see below).

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
├── requirements.txt    # packages required to run the application
└── README.md
```

## Running Locally
Clone the repo, create a virtual environment, and run the following (reference `.gitignore` for any files/directories which may need to be created manually after cloning):
```bash
# install dependencies
pip install -r requirements.txt
cd api
# run the API (requires model.pkl to exist - see/run notebooks/ to train one)
uvicorn main:app --reload
```

Then test the endpoint (should output a propensity_score and predicted_class):
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"NGIFTALL": 12, "CARDGIFT": 8, "AVGGIFT": 15.50, "MINRAMNT": 5.0, "MAXRAMNT": 25.0, "TIMELAG": 3.0, "MONTHS_SINCE_LAST_GIFT": 18, "MONTHS_SINCE_FIRST_GIFT": 91, "AGE": 62, "AGE_MISSING": 0, "INCOME": 4, "INCOME_MISSING": 0}'
```

Interactive API docs available at `http://127.0.0.1:8000/docs`.

## Azure ML Deployment

Registered the model to Azure ML Studio and deployed it as a managed real-time endpoint (`Standard_DS1_v2`, cost-optimized single instance), with a custom scoring script (`azure_scoring_script.py`) and a custom conda environment matching the exact local package versions (numpy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0, Python 3.12).

Getting to a working deployment involved diagnosing and resolving several infrastructure issues along the way:
- A subscription resource-provider registration gap (`Microsoft.PolicyInsights`) blocking endpoint provisioning.
- A pickle/numpy version mismatch between the local training environment and Azure's curated `sklearn-1.5` image (`numpy._core` didn't exist in the older numpy Azure shipped).
- A custom Docker-context environment that silently didn't apply its conda dependencies, traced to a missing `COPY`/`RUN conda env update` step.
- A legacy `azureml-defaults` dependency pulling in an incompatible old `pyarrow` build.

The final deployment was verified end-to-end. The following live request was made to the Azure endpoint from my local terminal: 

```bash
curl -X POST "https://donor-propensity-ws-zsujj.eastus.inference.ml.azure.com/score) \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <Key>" \   # actual bearer key omitted here
  -d "{\"data\": [{\"NGIFTALL\": 12, \"CARDGIFT\": 8, \"AVGGIFT\": 15.50, \"MINRAMNT\": 5.0, \"MAXRAMNT\": 25.0, \"TIMELAG\": 3.0, \"MONTHS_SINCE_LAST_GIFT\": 18, \"MONTHS_SINCE_FIRST_GIFT\": 91, \"AGE\": 62, \"AGE_MISSING\": 0, \"INCOME\": 4, \"INCOME_MISSING\": 0}]}"
```
Output: `[{\"propensity_score\": 0.5132, \"predicted_class\": 1}]`

Returned a propensity score of **0.5132** for a test donor — identical to the score returned by the local FastAPI endpoint for the same input, confirming the deployed model matches the locally validated one.

## Personal Takeaways

- **Simpler isn't always worse** — logistic regression outperformed both random forest and XGBoost on this dataset, a genuine, evidence-backed finding rather than a default assumption.
- **Threshold choice is a business decision, not just a statistical one** — the "right" cutoff depends on the real relative cost of a wasted contact vs. a missed donor.
- **Missing data needs a reason, not just a fill** — `TIMELAG`'s missing data was structural (no second gift), while `AGE`/`INCOME`'s was likely due to non-disclosure; each was handled differently as a result.
- **Deployment is rarely the smooth part** — getting a model from a notebook to a live, cloud-served endpoint surfaced several real environment and dependency mismatches, each requiring genuine debugging to resolve.

## Tech Stack

Python, pandas, scikit-learn, FastAPI, Azure ML

---
*Built as a self-directed learning project applying predictive modeling and deployment skills to a fundraising/advancement analytics context.*
