import pandas as pd

# loading the groundwater dataset
df = pd.read_csv("C:/programs/Groundwater_Forecasting/data/groundwater_final_dataset.csv")

# displaying the dataset information
print("Shape: ", df.shape)

print("\nColumns: ")
print(df.columns.tolist())

print("\nFirst 5 rows: ")
print(df.head())

print("\nMissing values: ")
print(df.isnull().sum())