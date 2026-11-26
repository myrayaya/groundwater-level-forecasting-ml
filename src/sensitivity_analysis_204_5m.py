
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "groundwater_final_dataset.csv"
PREDICTIONS_PATH = (
    BASE_DIR / "results" / "reviewer_revision" /
    "predictions" / "test_predictions.csv"
)
OUTPUT_PATH = (
    BASE_DIR / "results" / "reviewer_revision" /
    "tables" / "sensitivity_analysis_204_5m.csv"
)

TARGET = "Groundwater Level Quarterly Manual (meter)"

# Load datasets
df = pd.read_csv(DATA_PATH)
pred = pd.read_csv(PREDICTIONS_PATH)

df.columns = df.columns.str.strip()
pred.columns = pred.columns.str.strip()

# Find the unusual observation in the original dataset
affected = (
    df["Station"].astype(str).str.strip().eq("Shahbad 1(d)")
    & df["District"].astype(str).str.strip().str.upper().eq("KURUKSHETRA")
    & np.isclose(pd.to_numeric(df["GWL_Lag_1"], errors="coerce"), 204.5)
)

print("Affected observations in original dataset:", affected.sum())

if affected.sum() != 1:
    raise ValueError("Expected exactly one affected observation.")

print(df.loc[affected, [
    "Data Acquisition Time", "District", "Station",
    TARGET, "GWL_Lag_1"
]])

# Find the matching row in holdout predictions
excluded = (
    pred["Station"].astype(str).str.strip().eq("Shahbad 1(d)")
    & pred["District"].astype(str).str.strip().str.upper().eq("KURUKSHETRA")
    & np.isclose(pd.to_numeric(pred["Persistence"], errors="coerce"), 204.5)
)

print("\nAffected observations in holdout predictions:", excluded.sum())

if excluded.sum() != 1:
    raise ValueError(
        "Expected exactly one matching row in holdout_predictions.csv. "
        "Check the Station, District and Persistence values."
    )

# Actual and model prediction columns
MODEL_COLUMNS = {
    "Persistence": "Persistence",
    "Training Mean": "Training Mean",
    "Linear Regression": "Linear Regression",
    "Polynomial Regression": "Polynomial Regression",
    "Ridge Regression": "Ridge Regression",
    "KNN": "KNN",
    "MLP": "MLP"
}

# Calculate metrics on full and filtered holdout sets
full = pred.copy()
sensitivity = pred.loc[~excluded].copy()

print("\nFull holdout observations:", len(full))
print("Sensitivity holdout observations:", len(sensitivity))

results = []

for model, column in MODEL_COLUMNS.items():
    for label, subset in [
        ("Full", full),
        ("Excluding 204.5 observation", sensitivity)
    ]:
        actual = pd.to_numeric(subset["Actual"], errors="coerce")
        predicted = pd.to_numeric(subset[column], errors="coerce")

        valid = actual.notna() & predicted.notna()
        actual = actual[valid]
        predicted = predicted[valid]

        results.append({
            "Model": model,
            "Analysis": label,
            "N": len(actual),
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            "R2": r2_score(actual, predicted)
        })

# Save results
results_df = pd.DataFrame(results)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False)

print("\nSENSITIVITY ANALYSIS RESULTS")
print(results_df.round(3).to_string(index=False))
print(f"\nSaved to: {OUTPUT_PATH}")
