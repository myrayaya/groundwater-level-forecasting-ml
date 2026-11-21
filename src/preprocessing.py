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

def prepare_data(df):
    # converting date column
    df["Data Acquisition Time"] = pd.to_datetime(
        df["Data Acquisition Time"]
    )

    # sorting observations chronologically
    df = df.sort_values(
        "Data Acquisition Time"
    ).copy()

    # keeping the required modelling columns
    df = df[
        [
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
    ]

    return df

if __name__ == "__main__":
    df = pd.read_csv("C:/programs/Groundwater_Forecasting/data/groundwater_final_dataset.csv")

    df = prepare_data(df)

    print("Preprocessing completed.")
    print("Dataset Shape: ", df.shape)
    print("\nFeatures: ", FEATURES)
    print("\nTarget: ", TARGET)
    print("\nRecords by district: ")
    print(df["District"].value_counts())

    print("\nDate Range: ")
    print(df["Data Acquisition Time"].min(), "to", df["Data Acquisition Time"].max())     