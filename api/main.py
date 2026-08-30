"""
Donor Propensity Model - FastAPI Serving Endpoint
----------------------------------------------------
Loads a trained scikit-learn model (pickled) and exposes a /predict
endpoint that takes donor features and returns a propensity score.

Run locally with:
    uvicorn main:app --reload

Test with:
    curl -X POST http://127.0.0.1:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"recency_months": 6, "frequency": 4, "avg_gift_amount": 25.50, "age": 55, "years_since_first_gift": 8}'
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Donor Propensity Scoring API",
    description="Predicts likelihood of donor response/retention based on giving history.",
    version="1.0.0",
)

MODEL_PATH = Path("model.pkl")
model = None  # loaded on startup


@app.on_event("startup")
def load_model():
    """Load the trained model into memory once, at startup, not per-request."""
    global model
    if not MODEL_PATH.exists():
        # Don't crash the app on import - just log it. /predict will 503 until fixed.
        print(f"WARNING: {MODEL_PATH} not found. Train and save a model first.")
        return
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class DonorFeatures(BaseModel):
    """
    Feature set for a single donor, matching the 12 engineered features
    the model was actually trained on (see notebooks/eda.ipynb).
    """
    NGIFTALL: int = Field(..., description="Total lifetime number of gifts")
    CARDGIFT: int = Field(..., description="Lifetime gifts to card promotions")
    AVGGIFT: float = Field(..., description="Average gift amount in dollars")
    MINRAMNT: float = Field(..., description="Smallest gift amount to date")
    MAXRAMNT: float = Field(..., description="Largest gift amount to date")
    TIMELAG: float = Field(..., description="Months between first and second gift (0 if only one gift)")
    MONTHS_SINCE_LAST_GIFT: float = Field(..., description="Months since most recent gift, relative to reference mailing date")
    MONTHS_SINCE_FIRST_GIFT: float = Field(..., description="Donor tenure in months, relative to reference mailing date")
    AGE: float = Field(..., description="Donor age (imputed with median if originally missing)")
    AGE_MISSING: int = Field(..., description="1 if AGE was originally missing and imputed, else 0")
    INCOME: float = Field(..., description="Household income bracket, 1 (lowest) to 7 (highest)")
    INCOME_MISSING: int = Field(..., description="1 if INCOME was originally missing and imputed, else 0")

    class Config:
        json_schema_extra = {
            "example": {
                "NGIFTALL": 12,
                "CARDGIFT": 8,
                "AVGGIFT": 15.50,
                "MINRAMNT": 5.0,
                "MAXRAMNT": 25.0,
                "TIMELAG": 3.0,
                "MONTHS_SINCE_LAST_GIFT": 18,
                "MONTHS_SINCE_FIRST_GIFT": 91,
                "AGE": 62,
                "AGE_MISSING": 0,
                "INCOME": 4,
                "INCOME_MISSING": 0,
            }
        }


class PredictionResponse(BaseModel):
    propensity_score: float  # probability of response/retention, 0-1
    predicted_class: int     # 0 or 1 at a 0.5 threshold
    model_version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Donor Propensity API is running."}


@app.get("/health")
def health_check():
    """Basic health check - useful for uptime monitoring or load balancer probes."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: DonorFeatures):
    """
    Score a single donor. Returns a propensity probability and a
    binary prediction at a 0.5 threshold.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train and save model.pkl, then restart the API.",
        )

    # Convert the request into the same shape/column order the model was trained on.
    # IMPORTANT: this order must exactly match X_train's column order from the notebook.
    input_df = pd.DataFrame([{
        "NGIFTALL": features.NGIFTALL,
        "CARDGIFT": features.CARDGIFT,
        "AVGGIFT": features.AVGGIFT,
        "MINRAMNT": features.MINRAMNT,
        "MAXRAMNT": features.MAXRAMNT,
        "TIMELAG": features.TIMELAG,
        "MONTHS_SINCE_LAST_GIFT": features.MONTHS_SINCE_LAST_GIFT,
        "MONTHS_SINCE_FIRST_GIFT": features.MONTHS_SINCE_FIRST_GIFT,
        "AGE": features.AGE,
        "AGE_MISSING": features.AGE_MISSING,
        "INCOME": features.INCOME,
        "INCOME_MISSING": features.INCOME_MISSING,
    }])

    try:
        # Reorder columns to exactly match what the model saw during training.
        # More robust than hardcoding order by hand - sklearn remembers the
        # exact column order/names from .fit() via feature_names_in_.
        input_df = input_df[model.feature_names_in_]
        proba = model.predict_proba(input_df)[0][1]  # probability of class "1" (responder)
        predicted_class = int(proba >= 0.5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictionResponse(
        propensity_score=round(float(proba), 4),
        predicted_class=predicted_class,
    )


"""
"LOAD + DO EDA"

LOADING

import pandas as pd
df = pd.read_csv('cup98LRN.txt')
df.shape  # tells you (rows, columns) — sanity check it loaded right
df.head()  # prints the first 5 rows so you can eyeball what you're working with

####################################################################

DISTRIBUTIONS — for each column, "what values does it actually contain, and how often?" You're checking for weirdness before you trust the data.

df['TARGET_B'].value_counts()  # how many donors responded (1) vs didn't (0)?
df['AGE'].describe()  # min, max, mean, quartiles — catches garbage like age=0 or age=999
df['AGE'].hist()  # a quick histogram — is it roughly normal, skewed, bimodal?

You're looking for: obviously broken values (negative ages), weird outliers (someone donated $50,000 when everyone else gave $5–50), and whether a numeric column is actually behaving like a category in disguise.

####################################################################

MISSINGNESS — real-world data has gaps (a field wasn't collected for some records, a form was left blank). You need to know where, and how much, before you model anything.

df.isnull().sum()  # count of missing values per column
df.isnull().mean().sort_values(ascending=False)  # same thing as a percentage, sorted worst-first

If a column is 90% missing, you probably drop it. If it's 5% missing, you decide how to fill it in (mean, median, a placeholder category like "unknown") — that decision itself is a talking point for feature engineering.

####################################################################

CLASS BALANCE — specifically for your target column (TARGET_B), what fraction responded vs. didn't:

df['TARGET_B'].value_counts(normalize=True)


This matters a lot here: direct-mail response rates are typically ~5% (this dataset is a known example of imbalanced classification — way more non-responders than responders). If you don't account for that, your model can cheat by just predicting "no" every time and still look "accurate," which is exactly the kind of naive mistake this JD's evaluation criteria (precision/recall/AUC over plain accuracy) would catch. Noticing and explicitly addressing this imbalance is a good interview point.

In short: EDA is just "look before you leap" — a handful of .describe(), .value_counts(), and .isnull() calls, plus a few histograms, so you understand the data's shape and quality before you start engineering features and training models on top of it. You'll spend maybe 1–2 hours here, not the whole day.


FAST API ENDPOINTS:
A few things about how it's built:

Model loading happens once at startup, not per-request — a common beginner mistake is loading the pickle file inside the /predict function, which is slow and wasteful.
Pydantic schema (DonorFeatures) gives you automatic input validation and — bonus — free interactive API docs at http://127.0.0.1:8000/docs once it's running, which is genuinely useful to screen-share or link to in an interview.
/health endpoint is a small but real production-engineering touch (load balancers and uptime monitors ping this) — worth mentioning since it signals you're thinking about deployment, not just modeling.
The feature names (recency_months, frequency, etc.) are placeholders — swap them for whatever you land on during Day 2's feature engineering, just keep the column order in predict() matching how you trained the model.

To use it, you'll train your model in a notebook, then at the end save it with:

import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(trained_model, f)

and drop model.pkl next to main.py before running uvicorn main:app --reload.
"""