from pathlib import Path
import pandas as pd

# project paths
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "groundwater_final_dataset.csv"


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print("\nGROUNDWATER DATASET")
    print("-" * 40)
    print(f"Shape: {df.shape}")
    print(f"Unique stations: {df['Station'].nunique()}")
    print(f"Districts: {df['District'].unique().tolist()}")
    print(f"Date range: {df['Year'].min()} - {df['Year'].max()}")

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    return df


def main():
    df = load_dataset()


if __name__ == "__main__":
    main()
