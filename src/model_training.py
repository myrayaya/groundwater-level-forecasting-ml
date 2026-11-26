import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from preprocessing import prepare_data, FEATURES, TARGET
from models import get_models
from evaluation import evaluate_model


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "groundwater_final_dataset.csv"

GRAPH_PATH = ROOT / "results" / "graphs" / "model_evaluation"
PREDICTION_PATH = ROOT / "results" / "predictions"
TABLE_PATH = ROOT / "results" / "model_comparison.csv"


def chronological_split(df, train_ratio=0.8, gap_years=1):
   
    dates = sorted(df["Data Acquisition Time"].unique())

    split_index = int(len(dates) * train_ratio)
    split_date = pd.Timestamp(dates[split_index])

    train_end = split_date - pd.DateOffset(years=gap_years)

    train_df = df[df["Data Acquisition Time"] < train_end].copy()
    test_df = df[df["Data Acquisition Time"] >= split_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Chronological split produced an empty dataset.")

    return train_df, test_df, train_end, split_date


def main():

    # creating output directories
    GRAPH_PATH.mkdir(parents=True, exist_ok=True)
    PREDICTION_PATH.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # loading and preprocessing dataset
    df = pd.read_csv(DATA_PATH)
    df = prepare_data(df)

    # chronological train-test split
    train_df, test_df, train_end, split_date = chronological_split(df)

    X_train = train_df[FEATURES]
    X_test = test_df[FEATURES]

    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    print("Training records:", len(train_df))
    print("Testing records:", len(test_df))
    print("Training ends before:", train_end)
    print("Testing begins:", split_date)

    # loading models
    models = get_models()
    results = []
    prediction_data = test_df[
        ["Data Acquisition Time", "District", "Station", "Season", TARGET]
    ].copy()

    prediction_data.rename(columns={
        TARGET: "Actual",
        "Data Acquisition Time": "Date"
    }, inplace=True)

    # training, evaluating and generating predictions
    for name, model in models.items():

        print(f"\nTraining {name}")

        scores = evaluate_model(
            model, X_train, X_test, y_train, y_test
        )

        predictions = model.predict(X_test)
        residuals = y_test.to_numpy() - predictions

        filename = name.lower().replace(" ", "_")

        results.append({
            "Model": name,
            **scores
        })

        prediction_data[name] = predictions

        # actual vs predicted graph
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, predictions, alpha=0.6)

        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            linestyle="--",
            label="Perfect Prediction"
        )

        plt.title(f"Actual vs Predicted - {name}")
        plt.xlabel("Actual Groundwater Level (mbgl)")
        plt.ylabel("Predicted Groundwater Level (mbgl)")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            GRAPH_PATH / f"{filename}_scatter.png",
            dpi=300
        )
        plt.close()

        # residual graph
        plt.figure(figsize=(8, 6))
        plt.scatter(predictions, residuals, alpha=0.6)
        plt.axhline(y=0, linestyle="--", label="Zero Error Line")

        plt.title(f"Residual Analysis - {name}")
        plt.xlabel("Predicted Groundwater Level (mbgl)")
        plt.ylabel("Residuals (mbgl)")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            GRAPH_PATH / f"{filename}_residuals.png",
            dpi=300
        )
        plt.close()

    # saving model comparison
    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLE_PATH, index=False)

    # saving predictions
    prediction_data.to_csv(
        PREDICTION_PATH / "test_predictions.csv",
        index=False
    )

    print("\nMODEL COMPARISON")
    print(results_df.round(3).to_string(index=False))

    print("\nSaved model comparison:", TABLE_PATH)
    print("Saved predictions:", PREDICTION_PATH / "test_predictions.csv")


if __name__ == "__main__":
    main()
