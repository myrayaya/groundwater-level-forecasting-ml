import pandas as pd

TARGET = "Groundwater Level Quarterly Manual (meter)"

FEATURES = [
    "GWL_Lag_1",
    "GWL_Lag_2",
    "GWL_Lag_3",
    "Rainfall_prev_1m",
    "Rainfall_prev_3m",
    "Rainfall_prev_6m"
]

REQUIRED_COLUMNS = [
    "District",
    "Station",
    "Latitude",
    "Longitude",
    "Data Acquisition Time",
    "Year",
    "Month",
    "Season",
    TARGET
] + FEATURES


def prepare_data(df):
   
    df = df.copy()

    # validating required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # converting dates
    df["Data Acquisition Time"] = pd.to_datetime(
        df["Data Acquisition Time"],
        errors="coerce"
    )

    if df["Data Acquisition Time"].isna().any():
        raise ValueError("Invalid or missing dates found in dataset.")

    # sorting chronologically
    df = df.sort_values("Data Acquisition Time").reset_index(drop=True)

    # keeping modelling columns
    df = df[REQUIRED_COLUMNS].copy()

    # validating target and features
    if df[[TARGET] + FEATURES].isna().any().any():
        raise ValueError("Missing values found in target or model features.")

    return df


if __name__ == "__main__":
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent
    DATA_PATH = ROOT / "data" / "groundwater_final_dataset.csv"

    df = pd.read_csv(DATA_PATH)
    df = prepare_data(df)

    print("Preprocessing completed.")
    print("Dataset Shape:", df.shape)
    print("\nFeatures:", FEATURES)
    print("\nTarget:", TARGET)
    print("\nRecords by district:")
    print(df["District"].value_counts())
    print("\nDate Range:")
    print(df["Data Acquisition Time"].min(), "to",
          df["Data Acquisition Time"].max())
