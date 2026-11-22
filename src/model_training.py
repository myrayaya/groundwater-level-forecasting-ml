import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from preprocessing import prepare_data, FEATURES, TARGET
from models import get_models
from evaluation import evaluate_model

# loading the dataset 
df = pd.read_csv("C:/programs/Groundwater_Forecasting/data/groundwater_final_dataset.csv")

# preprocessing the data
df = prepare_data(df)

# sorting chronologically
df = df.sort_values("Data Acquisition Time").reset_index(drop=True)

# chronological 80:20 splitting
split_date = df["Data Acquisition Time"].sort_values().iloc[int(len(df) * 0.8)]

train_df = df[df["Data Acquisition Time"] < split_date]
test_df = df[df["Data Acquisition Time"] >= split_date]

X_train = train_df[FEATURES]
X_test = test_df[FEATURES]

y_train = train_df[TARGET]
y_test = test_df[TARGET]

# loading the models
models = get_models()
results = []
print("Models Loaded: ", models.keys())

# training and graph curation
for name, model in models.items():
    print(f"\nTraining {name}")

    # evaluating the model
    scores = evaluate_model(
        model, X_train, X_test, y_train, y_test
    )

    # prediction generation
    predictions = model.predict(X_test)

    # residual calculation
    residuals = y_test.values - predictions

    # file name standards
    filename = name.lower().replace(" ", "_")

    # graph-1 : actual vs predicted
    plt.figure(figsize = (8, 6))

    plt.scatter(y_test, predictions, alpha = 0.6)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle = "--",
        label = "Perfect Prediction"
    )

    plt.title(f"Actual Vs. Predicted - {name}")
    plt.xlabel("Actual Groundwater Level (mbgl)")
    plt.ylabel("Predicted Groundwater Level (mgbl)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"results/graphs/{filename}_scatter.png",
        dpi = 300
    )
    plt.close()

    # graph-2 : residual analysis
    plt.figure(figsize = (8, 6))

    plt.scatter(predictions, residuals, alpha = 0.6)

    plt.axhline(
        y = 0,
        linestyle = "--",
        label = "Zero Error Line"
    )

    plt.title(f"Residual Analysis - {name}")
    plt.xlabel("Predicted Groundwater Level (mbgl)")
    plt.ylabel("Residuals (mbgl)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"results/graphs/{filename}_residuals.png",
        dpi=300
    )

    plt.close()

    # storing model results
    results.append({
        "Model" : name, 
        **scores
    })

# model comparison
print("Number of results: ", len(results))
print("Results: ", results)
results_df = pd.DataFrame(results)

print("Training and testing split completed.")

print("\nTraining Records: ", len(X_train))
print("Testing Records: ", len(X_test))

print("\nTraining date range: ")
print(train_df["Data Acquisition Time"].min(), "to",
      train_df["Data Acquisition Time"].max())

print("\nTesting date range: ")
print(test_df["Data Acquisition Time"].min(), "to",
      test_df["Data Acquisition Time"].max())

print("\nModel Comparison:-")
print(results_df.to_string(index=False))

results_df.to_csv("results/model_comparison.csv", index=False)

# graph-3 : MAE comparison
plt.figure(figsize = (10, 6))
plt.bar(results_df["Model"], results_df["MAE"])

plt.title("Model Comparison - MAE")
plt.xlabel("Machine Learning Model")
plt.ylabel("MAE")
plt.xticks(rotation = 35, ha = "right")
plt.tight_layout()

plt.savefig(
    "results/graphs/mae_comparison.png",
    dpi = 300
)

plt.close()

# graph-4 : RMSE comparison
plt.figure(figsize = (10, 6))
plt.bar(results_df["Model"], results_df["RMSE"])

plt.title("Model Comparison - RMSE")
plt.xlabel("Machine Learning Model")
plt.ylabel("RMSE")
plt.xticks(rotation = 35, ha = "right")
plt.tight_layout()

plt.savefig(
    "results/graphs/rmse_comparison.png",
    dpi = 300
)

plt.close()

# graph-5 : r2 comparison
plt.figure(figsize= (10, 6))
plt.bar(results_df["Model"], results_df["R2"])

plt.title("Model Comparison - R2")
plt.xlabel("Machine Learning Model")
plt.ylabel("R2")
plt.xticks(rotation = 35, ha = "right")
plt.tight_layout()

plt.savefig(
    "results/graphs/r2_comparison.png",
    dpi = 300
)

plt.close()