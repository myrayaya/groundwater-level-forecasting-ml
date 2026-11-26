
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# path configuration

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "groundwater_final_dataset.csv"
OUTPUT_DIR = ROOT / "results" / "reviewer_revision"

TABLE_DIR = OUTPUT_DIR / "tables"
GRAPH_DIR = OUTPUT_DIR / "graphs"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

for folder in [TABLE_DIR, GRAPH_DIR, PREDICTION_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 407
N_SPLITS = 3
BOOTSTRAP_ITERATIONS = 1000

TARGET = "Groundwater Level Quarterly Manual (meter)"
DATE = "Data Acquisition Time"

GROUNDWATER_FEATURES = [
    "GWL_Lag_1",
    "GWL_Lag_2",
    "GWL_Lag_3"
]

RAINFALL_FEATURES = [
    "Rainfall_prev_1m",
    "Rainfall_prev_3m",
    "Rainfall_prev_6m"
]

FEATURES = GROUNDWATER_FEATURES + RAINFALL_FEATURES


# loading and preparing dataset

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Check that groundwater_final_dataset.csv is inside data/."
    )

df = pd.read_csv(DATA_PATH)

required_columns = [TARGET, DATE] + FEATURES + [
    "District", "Station", "Latitude", "Longitude", "Season"
]

missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# save missing-value report before filtering
missing_report = pd.DataFrame({
    "Column": df.columns,
    "Missing_Count": df.isna().sum().values,
    "Missing_Percentage": (
        df.isna().mean().values * 100
    )
})

missing_report.to_csv(
    TABLE_DIR / "missing_value_report.csv",
    index=False
)

df[DATE] = pd.to_datetime(df[DATE], errors="coerce")

for col in FEATURES + [TARGET]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=[DATE, TARGET] + FEATURES
).copy()

df = df.sort_values(DATE).reset_index(drop=True)

# unique monitoring-well identifier
df["Well_ID"] = (
    df["District"].astype(str) + "_" +
    df["Station"].astype(str) + "_" +
    df["Latitude"].astype(str) + "_" +
    df["Longitude"].astype(str)
)

print("Dataset shape:", df.shape)
print("Unique wells:", df["Well_ID"].nunique())
print("Date range:", df[DATE].min(), "to", df[DATE].max())


# train-test split

if len(df) <= TEST_SIZE:
    raise ValueError("Dataset is too small for the requested test size.")

train = df.iloc[:-TEST_SIZE].copy()
test = df.iloc[-TEST_SIZE:].copy()

print("\nChecking train-test date overlap:")
print("Training end:", train[DATE].max())
print("Testing start:", test[DATE].min())

overlap = set(train[DATE]).intersection(set(test[DATE]))
print("Overlapping dates:", len(overlap))

X_train = train[FEATURES]
X_test = test[FEATURES]

y_train = train[TARGET]
y_test = test[TARGET]

print("\nTrain observations:", len(train))
print("Test observations:", len(test))
print("Training end:", train[DATE].max())
print("Testing start:", test[DATE].min())


# model building

def build_model(model_name, parameters):

    if model_name == "Linear Regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])

    elif model_name == "Polynomial Regression":
        return Pipeline([
            (
                "poly",
                PolynomialFeatures(
                    degree=parameters["degree"],
                    include_bias=False
                )
            ),
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])

    elif model_name == "Ridge Regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=parameters["alpha"]))
        ])

    elif model_name == "KNN":
        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                KNeighborsRegressor(
                    n_neighbors=parameters["n_neighbors"]
                )
            )
        ])

    elif model_name == "MLP":
        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=parameters["hidden_layer_sizes"],
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    early_stopping=False
                )
            )
        ])

    raise ValueError(f"Unknown model: {model_name}")


# small grids to keep the experiment manageable
MODEL_GRIDS = {
    "Linear Regression": [{}],

    "Polynomial Regression": [
        {"degree": 2},
        {"degree": 3}
    ],

    "Ridge Regression": [
        {"alpha": 0.1},
        {"alpha": 1.0},
        {"alpha": 10.0}
    ],

    "KNN": [
        {"n_neighbors": 3},
        {"n_neighbors": 5},
        {"n_neighbors": 7}
    ],

    "MLP": [
        {"hidden_layer_sizes": (50,)},
        {"hidden_layer_sizes": (100, 50)}
    ]
}


# cross-validation

def create_date_folds(training_data):

    unique_dates = np.sort(
        training_data[DATE].dropna().unique()
    )

    if len(unique_dates) <= N_SPLITS:
        raise ValueError("Not enough unique dates for CV.")

    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    folds = []

    for train_idx, val_idx in splitter.split(unique_dates):

        train_dates = unique_dates[train_idx]
        val_dates = unique_dates[val_idx]

        fold_train = np.flatnonzero(
            training_data[DATE].isin(train_dates).values
        )

        fold_val = np.flatnonzero(
            training_data[DATE].isin(val_dates).values
        )

        folds.append((fold_train, fold_val))

    return folds


folds = create_date_folds(train)

print("\nNumber of chronological folds:", len(folds))


def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    }


# hyperparameter training

cv_results = []
best_models = {}

for model_name, parameter_grid in MODEL_GRIDS.items():

    best_rmse = float("inf")
    best_parameters = None
    best_fold_scores = None

    print(f"\nTuning {model_name}...")

    for parameters in parameter_grid:

        fold_scores = []

        for fold_number, (train_idx, val_idx) in enumerate(folds, 1):

            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]

            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            model = build_model(model_name, parameters)

            model.fit(X_fold_train, y_fold_train)

            predictions = model.predict(X_fold_val)

            metrics = calculate_metrics(
                y_fold_val,
                predictions
            )

            fold_scores.append(metrics)

        mean_rmse = np.mean([
            score["RMSE"] for score in fold_scores
        ])

        mean_mae = np.mean([
            score["MAE"] for score in fold_scores
        ])

        mean_r2 = np.mean([
            score["R2"] for score in fold_scores
        ])

        std_rmse = np.std([
            score["RMSE"] for score in fold_scores
        ], ddof=1) if len(fold_scores) > 1 else 0

        cv_results.append({
            "Model": model_name,
            "Parameters": str(parameters),
            "Mean_RMSE": mean_rmse,
            "Std_RMSE": std_rmse,
            "Mean_MAE": mean_mae,
            "Mean_R2": mean_r2
        })

        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_parameters = parameters
            best_fold_scores = fold_scores

    best_models[model_name] = {
        "parameters": best_parameters,
        "cv_rmse": best_rmse,
        "fold_scores": best_fold_scores
    }

    print("Best parameters:", best_parameters)
    print("Mean validation RMSE:", round(best_rmse, 4))


cv_df = pd.DataFrame(cv_results)

cv_df.to_csv(
    TABLE_DIR / "chronological_cv_results.csv",
    index=False
)


# final test evaluation

test_results = []
test_predictions = pd.DataFrame({
    "Actual": y_test.values,
    "Date": test[DATE].values,
    "District": test["District"].values,
    "Station": test["Station"].values,
    "Season": test["Season"].values,
    "Well_ID": test["Well_ID"].values
})

fitted_models = {}

# Baseline 1: persistence
persistence_predictions = test["GWL_Lag_1"].values

# Baseline 2: training mean
mean_predictions = np.full(
    len(test),
    y_train.mean()
)

baseline_predictions = {
    "Persistence": persistence_predictions,
    "Training Mean": mean_predictions
}

for name, predictions in baseline_predictions.items():

    metrics = calculate_metrics(y_test, predictions)

    test_results.append({
        "Model": name,
        **metrics
    })

    test_predictions[name] = predictions


# train each tuned model on the complete training set
for model_name, result in best_models.items():

    parameters = result["parameters"]

    model = build_model(model_name, parameters)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = calculate_metrics(y_test, predictions)

    test_results.append({
        "Model": model_name,
        **metrics
    })

    test_predictions[model_name] = predictions
    fitted_models[model_name] = model

    print(
        model_name,
        "MAE:", round(metrics["MAE"], 4),
        "RMSE:", round(metrics["RMSE"], 4),
        "R2:", round(metrics["R2"], 4)
    )


test_results_df = pd.DataFrame(test_results)

test_results_df.to_csv(
    TABLE_DIR / "final_test_comparison.csv",
    index=False
)

test_predictions.to_csv(
    PREDICTION_DIR / "test_predictions.csv",
    index=False
)


# selecting best model based on cross-validation

best_model_name = min(
    best_models,
    key=lambda name: best_models[name]["cv_rmse"]
)

best_model = fitted_models[best_model_name]
best_parameters = best_models[best_model_name]["parameters"]

print("\nBest model selected using CV:", best_model_name)
print("Selected parameters:", best_parameters)


# error analysis

best_predictions = best_model.predict(X_test)

error_df = test[
    ["District", "Station", "Season", "Well_ID", DATE, TARGET]
].copy()

error_df["Predicted"] = best_predictions
error_df["Residual"] = (
    error_df[TARGET] - error_df["Predicted"]
)
error_df["Absolute_Error"] = error_df["Residual"].abs()

# deep-well threshold based only on training targets
deep_threshold = y_train.quantile(0.75)

error_df["Depth_Group"] = np.where(
    error_df[TARGET] >= deep_threshold,
    "Deep wells",
    "Other wells"
)


def grouped_error_analysis(data, group_column):

    records = []

    for group, subset in data.groupby(group_column):

        if len(subset) == 0:
            continue

        metrics = calculate_metrics(
            subset[TARGET],
            subset["Predicted"]
        )

        records.append({
            group_column: group,
            "Observations": len(subset),
            **metrics
        })

    return pd.DataFrame(records)


district_errors = grouped_error_analysis(
    error_df, "District"
)

season_errors = grouped_error_analysis(
    error_df, "Season"
)

depth_errors = grouped_error_analysis(
    error_df, "Depth_Group"
)

district_errors.to_csv(
    TABLE_DIR / "district_error_analysis.csv",
    index=False
)

season_errors.to_csv(
    TABLE_DIR / "season_error_analysis.csv",
    index=False
)

depth_errors.to_csv(
    TABLE_DIR / "deep_well_error_analysis.csv",
    index=False
)

# Largest prediction errors
largest_errors = error_df.sort_values(
    "Absolute_Error",
    ascending=False
).head(20)

largest_errors.to_csv(
    TABLE_DIR / "largest_prediction_errors.csv",
    index=False
)

error_df.to_csv(
    PREDICTION_DIR / "best_model_error_analysis.csv",
    index=False
)

print("\nDeep-well threshold:", round(deep_threshold, 4), "mbgl")


# feature ablation

feature_groups = {
    "Full Features": FEATURES,
    "Without Rainfall": GROUNDWATER_FEATURES,
    "Without Groundwater Lags": RAINFALL_FEATURES
}

ablation_results = []

for experiment_name, selected_features in feature_groups.items():

    model = build_model(best_model_name, best_parameters)

    model.fit(
        train[selected_features],
        y_train
    )

    predictions = model.predict(
        test[selected_features]
    )

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    ablation_results.append({
        "Experiment": experiment_name,
        "Features": ", ".join(selected_features),
        **metrics
    })


ablation_df = pd.DataFrame(ablation_results)

ablation_df.to_csv(
    TABLE_DIR / "feature_ablation_results.csv",
    index=False
)


# well-level bootstrap confidence intervals

def cluster_bootstrap(
    actual,
    predicted,
    well_ids,
    iterations=BOOTSTRAP_ITERATIONS
):

    bootstrap_results = []

    unique_wells = np.unique(well_ids)
    rng = np.random.default_rng(RANDOM_STATE)

    for _ in range(iterations):

        sampled_wells = rng.choice(
            unique_wells,
            size=len(unique_wells),
            replace=True
        )

        sampled_indices = []

        for well in sampled_wells:
            sampled_indices.extend(
                np.flatnonzero(well_ids == well)
            )

        sampled_indices = np.asarray(sampled_indices)

        if len(np.unique(actual[sampled_indices])) < 2:
            continue

        metrics = calculate_metrics(
            actual[sampled_indices],
            predicted[sampled_indices]
        )

        bootstrap_results.append(metrics)

    bootstrap_df = pd.DataFrame(bootstrap_results)

    intervals = []

    for metric in ["MAE", "RMSE", "R2"]:

        intervals.append({
            "Metric": metric,
            "Lower_95_CI": bootstrap_df[metric].quantile(0.025),
            "Upper_95_CI": bootstrap_df[metric].quantile(0.975),
            "Bootstrap_Mean": bootstrap_df[metric].mean()
        })

    return pd.DataFrame(intervals)


bootstrap_df = cluster_bootstrap(
    y_test.to_numpy(),
    best_predictions,
    test["Well_ID"].to_numpy()
)

bootstrap_df.to_csv(
    TABLE_DIR / "bootstrap_confidence_intervals.csv",
    index=False
)


# distribution graphs

plt.figure(figsize=(8, 5))
plt.hist(df[TARGET], bins=30, edgecolor="black")
plt.xlabel("Groundwater Level (mbgl)")
plt.ylabel("Frequency")
plt.title("Distribution of Groundwater Levels")
plt.tight_layout()
plt.savefig(
    GRAPH_DIR / "groundwater_distribution.png",
    dpi=300
)
plt.close()


plt.figure(figsize=(8, 5))
plt.hist(
    df["Rainfall_prev_1m"],
    bins=30,
    edgecolor="black"
)
plt.xlabel("Previous Month Rainfall")
plt.ylabel("Frequency")
plt.title("Distribution of Previous Month Rainfall")
plt.tight_layout()
plt.savefig(
    GRAPH_DIR / "rainfall_distribution.png",
    dpi=300
)
plt.close()


# actual  vs. predicted

plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.6
)

minimum = min(y_test.min(), best_predictions.min())
maximum = max(y_test.max(), best_predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Groundwater Level (mbgl)")
plt.ylabel("Predicted Groundwater Level (mbgl)")
plt.title(f"Actual vs Predicted: {best_model_name}")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "actual_vs_predicted.png",
    dpi=300
)
plt.close()


# residual distribution

plt.figure(figsize=(8, 5))

plt.hist(
    error_df["Residual"],
    bins=30,
    edgecolor="black"
)

plt.axvline(0, linestyle="--")

plt.xlabel("Residual (Actual - Predicted) in metres")
plt.ylabel("Frequency")
plt.title("Prediction Residual Distribution")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "residual_distribution.png",
    dpi=300
)
plt.close()


# model comparison graph

plt.figure(figsize=(10, 5))

plt.bar(
    test_results_df["Model"],
    test_results_df["RMSE"]
)

plt.ylabel("RMSE (metres)")
plt.xlabel("Model")
plt.title("Test-Set RMSE Comparison")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "model_rmse_comparison.png",
    dpi=300
)
plt.close()


# feature ablation graph

plt.figure(figsize=(8, 5))

plt.bar(
    ablation_df["Experiment"],
    ablation_df["RMSE"]
)

plt.ylabel("RMSE (metres)")
plt.title("Feature Ablation: RMSE Comparison")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "feature_ablation.png",
    dpi=300
)
plt.close()


# saving experiment data summary

summary = {
    "Dataset observations": len(df),
    "Training observations": len(train),
    "Testing observations": len(test),
    "Unique wells": df["Well_ID"].nunique(),
    "Training start": str(train[DATE].min()),
    "Training end": str(train[DATE].max()),
    "Testing start": str(test[DATE].min()),
    "Testing end": str(test[DATE].max()),
    "Number of CV folds": len(folds),
    "Best model by CV RMSE": best_model_name,
    "Best model parameters": str(best_parameters),
    "Deep well threshold (mbgl)": float(deep_threshold)
}

pd.DataFrame(
    list(summary.items()),
    columns=["Item", "Value"]
).to_csv(
    TABLE_DIR / "experiment_summary.csv",
    index=False
)

print("REVIEWER EXPERIMENTS COMPLETED")

print("\nBest model:", best_model_name)

print("\nTest-set comparison:")
print(test_results_df.round(4).to_string(index=False))

print("\nAblation results:")
print(ablation_df.round(4).to_string(index=False))

print("\nBootstrap confidence intervals:")
print(bootstrap_df.round(4).to_string(index=False))

print("\nAll outputs saved in:")
print(OUTPUT_DIR)
