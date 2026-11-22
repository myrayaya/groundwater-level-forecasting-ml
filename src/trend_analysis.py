import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# loading the historical groundwater dataset
df = pd.read_csv("C:/programs/Groundwater_Forecasting/data/gwl_manual_quarterly_cgwb_hr_1991_2025.csv")

# converting acquisition date into datetime format
df["Data Acquisition Time"] = pd.to_datetime(
    df["Data Acquisition Time"],
    format = "%d-%m-%Y %H:%M",
    errors = "coerce"
)

# keeping only Karnal and Kurukshetra
df["District"] = df["District"].str.strip().str.upper()

df = df[df["District"].isin(["KARNAL", "KURUKSHETRA"])].copy()

# removing rows with missing groundwater levels or dates
target = "Groundwater Level Quarterly Manual (meter)"
df = df.dropna(subset=["Data Acquisition Time", target])

# display information
print("\nGroundwater Trend Analysis:- ")

print("\nTotal records: ", len(df))
print(df["District"].value_counts())

print("\nDate range:- ")
print(df["Data Acquisition Time"].min())
print("to")
print(df["Data Acquisition Time"].max())

print("\nUnique stations by district:- ")
print(
    df.groupby("District")["Station"].unique()
)

# identifying the seasons using months
df["Month"] = df["Data Acquisition Time"].dt.month
df["Season"] = df["Month"].map({
    5 : "Pre-Monsoon",
    1 : "Post-Monsoon",
    11 : "Post-Monsoon"
})

# keeping only pre-monsoon and post-monsoon observations
seasonal_df = df.dropna(subset=["Season"]).copy()

# identifying each individual monitoring well
well_columns = [
    "District",
    "Station",
    "Latitude",
    "Longitude"
]

# counting the observations for each well in each season
station_counts = (
    seasonal_df
    .groupby(well_columns + ["Season"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# displaying results
print("\nSeasonal Records:-")
print(seasonal_df["Season"].value_counts())

print("\nUnique wells by district:-")
print(
    df.groupby("District")[well_columns[1:]].apply(
        lambda x: x.drop_duplicates().shape[0]
    )
)

print("\nSeasonal Observations:- ")
print(station_counts.head(10))

print("\nTotal Unique Wells: ", len(station_counts))

print("\nChecking Season Classification:- ")
print(seasonal_df["Season"].value_counts(dropna=False))
print("\nSeason values: ", seasonal_df["Season"].unique())

# function to calculate groundwater trend
def calculate_trend(group):
    group = group.sort_values("Data Acquisition Time")

    observations = len(group)

    years = (
        group["Data Acquisition Time"].dt.year
        + (group["Data Acquisition Time"].dt.dayofyear - 1) / 365.25
    )

    # avoiding regression for insufficient data
    if observations < 5 or years.nunique() < 2:
        return pd.Series({
            "Observations" : observations,
            "Slope" : float("nan"),
            "R2" : float("nan")
        })

    groundwater_levels = group[target]

    slope, intercept, r_value, p_value, std_err = linregress(
        years, groundwater_levels
    )

    return pd.Series({
        "Observations" : observations,
        "Slope" : slope,
        "R2" : r_value ** 2
    })

# calculating trends separately for every well and season
trend_results = (
    seasonal_df
    .groupby(well_columns + ["Season"])
    .apply(calculate_trend)
    .reset_index()
)

# classifying reliability
def classify_reliability (count):
    if count >= 10:
        return "Reliable"
    elif count >= 5:
        return "Limited"
    else:
        return "Insufficient"

trend_results["Reliability"] = trend_results["Observations"].apply(
    classify_reliability
)

# classifying groundwater trends
def classify_trends(slope):
    if slope > 0.05:
        return "Declining"
    elif slope < -0.05:
        return "Improving"
    else:
        return "Stable"

trend_results["Trend"] = trend_results.apply(
    lambda row: (
        "Insufficient Data"
        if row["Reliability"] == "Insufficient" or pd.isna(row["Slope"])
        else classify_trends(row["Slope"])
    ),
    axis=1
)

# ensuring trend classification is not applied to insufficient data
trend_results.loc[
    trend_results["Reliability"] == "Insufficient", "Trend"
] = "Insufficient Data"

# displaying results
print("Groundwater Trend Results:- ")
print(trend_results.head(10))

print("\nReliability Summary: ")
print(trend_results.groupby(["District", "Season"])["Reliability"].value_counts())

print("\nTrend Summary: ")
print(trend_results.groupby(["District", "Season"])["Trend"].value_counts())

# graph-1 : trend distribution by District
plot_data = trend_results[
    trend_results["Trend"] != "Insufficient Data"
]

plt.figure(figsize = (10, 6))

sns.countplot(
    data = plot_data,
    x = "District",
    hue = "Trend", 
    order = ["KARNAL", "KURUKSHETRA"],
)

plt.title("Groundwater Trend Distribution by District")
plt.xlabel("District")
plt.ylabel("Number of Wells")
plt.legend(title = "Groundwater Trend")
plt.tight_layout()

plt.savefig("results/graphs/groundwater_trend_distribution.png")
plt.show()
plt.close()

# graph-2 : data reliability by district
plt.figure(figsize = (10, 6))

sns.countplot(
    data = trend_results,
    x = "District",
    hue = "Reliability",
    order = ["KARNAL", "KURUKSHETRA"],
    hue_order = ["Reliable", "Limited", "Insufficient"]
)

plt.title("Groundwater Data Reliability by District")
plt.xlabel("District")
plt.ylabel("Number of Station-Season Records")
plt.legend(title = "Data Reliability")
plt.tight_layout()

plt.savefig("results/graphs/groundwater_reliability_distribution.png")
plt.show()
plt.close()

# graph-3 : seasonal trend comparison
g = sns.catplot(
    data = plot_data,
    x = "District",
    hue = "Trend",
    col = "Season",
    kind = "count",
    order = ["KARNAL", "KURUKSHETRA"],
    hue_order = ["Declining", "Improving", "Stable"],
    height = 5,
    aspect = 1.1
)

g.set_axis_labels("District", "Number of Wells")
g.set_titles("{col_name}")
g.fig.suptitle("Seasonal Groundwater Trend Comparison", y=1.05)

g.savefig("results/graphs/seasonal_groundwater_trends.png")
plt.show()
plt.close("all")

# graph-4 : station-wise groundwater levels
selected_stations = {
    "KARNAL" : ["Kalri Jagir S", "Amin S"],
    "KURUKSHETRA" : ["Pehowa (D)"]
}

for district, stations in selected_stations.items():

    for station in stations:

        station_data = df[
            (df["District"] == district) &
            (df["Station"].str.strip() == station)
        ].copy()

        if station_data.empty:
            print(f"No data found for {station} in {district}")
            continue

        station_data = station_data.sort_values(
            "Data Acquisition Time"
        )

        plt.figure(figsize = (11, 5))

        sns.lineplot(
            data = station_data,
            x = "Data Acquisition Time",
            y = target,
            marker = "o"
        )

        plt.title(
            f"Groundwater Level Changes - {station}, {district}"
        )

        plt.xlabel("Year")
        plt.ylabel("Groundwater Depth (mbgl)")
        plt.xticks(rotation = 45)
        plt.tight_layout()

        filename = {
            f"{district.lower()}_"
            f"{station.lower().replace(" ", "_").replace("(", "").replace(")", "")}"
            f"_groundwater_levels.png"
        }

        plt.savefig(f"results/graphs/{filename}", format = "png")
        plt.show()
        plt.close()
        print(f"Graph saved: {filename}")