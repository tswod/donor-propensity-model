import json
import pickle
import os
import pandas as pd

# Column order/names must exactly match what the model was trained on.
FEATURE_COLUMNS = [
    "NGIFTALL", "CARDGIFT", "AVGGIFT", "MINRAMNT", "MAXRAMNT", "TIMELAG",
    "MONTHS_SINCE_LAST_GIFT", "MONTHS_SINCE_FIRST_GIFT",
    "AGE", "AGE_MISSING", "INCOME", "INCOME_MISSING",
]

model = None


def init():
    """
    Called once when the endpoint starts up. Loads the registered model
    into memory. AZUREML_MODEL_DIR is an environment variable Azure sets
    automatically, pointing to wherever it placed your registered model files.
    """
    global model
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    # Azure nests the registered file under a versioned subfolder - this finds
    # model.pkl wherever it actually landed rather than hardcoding a deep path.
    model_path = None
    for root, dirs, files in os.walk(model_dir):
        if "model.pkl" in files:
            model_path = os.path.join(root, "model.pkl")
            break

    if model_path is None:
        raise FileNotFoundError(f"model.pkl not found under {model_dir}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)


def run(raw_data):
    """
    Called on every prediction request. `raw_data` is a JSON string;
    expected format: {"data": [{...one donor's features...}, ...]}
    Supports batching multiple donors in one request.
    """
    try:
        body = json.loads(raw_data)
        records = body["data"]  # a list of donor feature dicts

        input_df = pd.DataFrame(records)
        # Reorder/select columns to exactly match training - same pattern as main.py
        input_df = input_df[model.feature_names_in_]

        probabilities = model.predict_proba(input_df)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        results = [
            {"propensity_score": round(float(p), 4), "predicted_class": int(c)}
            for p, c in zip(probabilities, predictions)
        ]
        return json.dumps(results)

    except Exception as e:
        return json.dumps({"error": str(e)})