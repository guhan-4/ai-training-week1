"""Score a CSV of new Telco customers with outputs/final_model.joblib.

Run:
  python outputs/predict_new.py --input data/new_customers.csv

The input should contain the same predictor columns used to train the model;
customerID is optional and is copied into the predictions output when present.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV of new customer rows")
    parser.add_argument("--model", type=Path, default=Path("outputs/final_model.joblib"))
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.csv"))
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    threshold = float(bundle["threshold"])
    feature_columns = bundle["feature_columns"]
    new_data = pd.read_csv(args.input)
    ids = new_data["customerID"].copy() if "customerID" in new_data else pd.Series(new_data.index, name="row_id")
    new_data = new_data.drop(columns=[bundle.get("target_column", "Churn"), "customerID"], errors="ignore")

    missing = [column for column in feature_columns if column not in new_data.columns]
    if missing:
        raise ValueError(
            "Input is missing required model columns: " + ", ".join(missing)
            + ". Provide the same feature columns as the training dataset."
        )
    new_data = new_data.loc[:, feature_columns].copy()
    if "TotalCharges" in new_data.columns:
        new_data["TotalCharges"] = pd.to_numeric(
            new_data["TotalCharges"].replace(r"^\s*$", np.nan, regex=True), errors="coerce"
        )

    probability = model.predict_proba(new_data)[:, 1]
    prediction = (probability >= threshold).astype(int)
    output = pd.DataFrame({
        "customerID": ids.to_numpy(),
        "churn_probability": probability,
        "predicted_churn": prediction,
        "decision_threshold": threshold,
    })
    output["predicted_churn_label"] = np.where(output["predicted_churn"].eq(1), "Yes", "No")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Scored {len(output)} customers; saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
