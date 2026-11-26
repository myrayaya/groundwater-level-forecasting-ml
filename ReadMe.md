# Groundwater-Level Forecasting and Trend Analysis

A machine-learning project for analysing historical groundwater-level behaviour and evaluating regression models for groundwater-depth prediction in **Karnal and Kurukshetra, Haryana, India**.

> **Interpretation:** Groundwater level is recorded as depth below ground level (mbgl). A larger depth value means the water table is farther below the ground surface. A positive long-term slope therefore indicates a declining groundwater-level trend.

## Project objectives

- Prepare seasonal groundwater observations for Karnal and Kurukshetra.
- Combine groundwater observations with antecedent rainfall and previous groundwater measurements.
- Train and compare five regression models.
- Evaluate predictions using MAE, RMSE and R² on a chronological holdout set.
- Independently analyse station-wise pre-monsoon and post-monsoon groundwater trends and data sufficiency.
- Visualise district-level trends, monitoring-data reliability, selected station histories and model performance.

## Study area and data

The study focuses on two districts in Haryana: **Karnal** and **Kurukshetra**.

The groundwater source is the Central Ground Water Board (CGWB) / India-WRIS groundwater monitoring data. The rainfall features were derived from the India Meteorological Department (IMD) daily gridded rainfall product at 0.25° spatial resolution. For each monitoring-well location, the nearest grid cell was used and daily rainfall was aggregated to monthly totals.

The processed seasonal modelling dataset contains **1,882 observations**:

- Karnal: 1,282 observations (577 pre-monsoon; 705 post-monsoon).
- Kurukshetra: 600 observations (298 pre-monsoon; 302 post-monsoon).

The modelling dataset covers **15 May 1997 to 1 November 2023**. The selected target observations are May (pre-monsoon) and November (post-monsoon). The source records are irregularly spaced; lag features represent previous available observations for the same monitoring well and should not be interpreted as guaranteed consecutive calendar quarters.

### Modelling variables

**Target:** `Groundwater Level Quarterly Manual (meter)`

**Predictors:**

- `GWL_Lag_1`, `GWL_Lag_2`, `GWL_Lag_3`
- `Rainfall_prev_1m`, `Rainfall_prev_3m`, `Rainfall_prev_6m`

The groundwater lags are calculated within each unique well, identified using district, station name, latitude and longitude. Rainfall is matched to the well location using the nearest IMD grid cell.

## Methodology

### Preprocessing

1. Parse observation dates and sort measurements chronologically within each well.
2. Retain the study districts and selected seasonal observations.
3. Derive monthly rainfall totals and antecedent rainfall windows.
4. Construct three previous-observation groundwater lag features per well.
5. Remove records that lack required predictor or target values.
6. Split the observations chronologically into training and testing sets, with 1,475 training observations and 407 testing observations. The split maintains temporal ordering and avoids overlap between training and testing dates.

### Regression models

1. Linear Regression with Lags
2. Polynomial Regression (degree 2)
3. Ridge Regression
4. K-Nearest Neighbours (KNN) Regressor
5. Multi-Layer Perceptron (MLP) Regressor

Feature scaling is used in the model pipelines. The MLP configuration used in the reported run has hidden layers of 100 and 50 units, with a maximum of 2,000 iterations and a fixed random seed.

### Evaluation metrics

- **MAE:** average absolute prediction error; lower is better.
- **RMSE:** square root of mean squared error; lower is better and larger errors receive more weight.
- **R²:** coefficient of determination; higher is generally better. R² is not a percentage accuracy measure.

## Model results

| Model | MAE (m) | RMSE (m) | R² |
|---|---:|---:|---:|
| Linear Regression with Lags | 2.208 | 3.210 | 0.9222 |\n| Polynomial Regression | 2.095 | 3.317 | 0.9170 |\n| Ridge Regression | 2.206 | 3.210 | 0.9222 |\n| KNN Regressor | 2.295 | 3.329 | 0.9163 |\n| MLP Regressor | 1.961 | 2.978 | 0.9331 |\n

The MLP Regressor had the lowest MAE (1.961 m), lowest RMSE (2.978 m) and highest R² (0.9331) in this particular holdout evaluation. This indicates the strongest performance among the five tested models on this test set; it does **not** establish that MLP will be best on other districts, time periods or unseen wells.

## Historical station-wise trend analysis

Trend analysis is a separate component from the machine-learning prediction experiment. It uses station-season observations to fit a linear trend against time.

- Pre-monsoon: May.
- Post-monsoon: January and November.
- Reliability: Reliable = at least 10 observations; Limited = 5–9; Insufficient = fewer than 5.
- Trend classification uses the slope threshold of ±0.05 m/year: positive above the threshold = Declining; negative below the threshold = Improving; otherwise Stable.
- Station records are distinguished using district, station, latitude and longitude.

A positive slope means groundwater depth is increasing over time, which corresponds to a deeper water table. Trend classifications are exploratory screening results, not official CGWB groundwater-resource categories.

## Sensitivity Analysis

To reproduce the sensitivity analysis of the 204.5 m
lagged groundwater-level observation, first run:

```text
python src/reviewer_experiments.py
```

Then run:

```text
python src/sensitivity_analysis_204_5m.py
```

The script compares all seven models using the complete
chronological holdout and a second evaluation excluding
the affected Shahbad 1(d), Kurukshetra observation.

It also checks the occurrence of the 204.5 m value in
the training, cross-validation and test partitions.

Results are saved in:
results/reviewer_revision/tables/

## Repository outputs

Suggested organisation:

```text
.
├── data/
│   ├── groundwater_final_dataset.csv
│   └── gwl_manual_quarterly_cgwb_hr_1991_2025.csv
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── model_training.py
│   ├── trend_analysis.py
│   ├── reviewer_experiments.py
│   └── sensitivity_analysis_204_5m.py
├── results/
│   ├── model_comparison.csv
│   ├── groundwater_trend_results.csv
│   ├── graphs/
│   └── reviewer_revision/
├── main.py
├── requirements.txt
└── ReadMe.md
```

Large raw source datasets are not included here by default. Follow the source links below and observe the data provider's terms before redistributing source data.

## Run locally

Python 3.10+ is recommended.

```bash
git clone https://github.com/myrayaya/groundwater-level-forecasting-ml.git
cd groundwater-level-forecasting-ml

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Check the file paths in the scripts before running. The final processed dataset and any required rainfall data must be placed in the expected `data/` directory.

### Execute the project

Run the following scripts from the project root:

```bash
python main.py
python src/preprocessing.py
python src/trend_analysis.py
python src/model_training.py
python src/reviewer_experiments.py
python src/sensitivity_analysis_204_5m.py
```

## Limitations

- Groundwater monitoring observations are irregular in time. A previous-observation lag is not necessarily a one-quarter lag.
- The chronological split evaluates later observations from the overall set of monitoring wells; it is not a fully independent test on unseen wells.
- The global time split may leave a gap between the latest training date and earliest test date because observations are irregular.
- The trend slope is a simple linear summary and does not model all hydrogeological, pumping, land-use or recharge influences.
- Rainfall is only one influence on groundwater levels. The models do not establish causal effects.
- Results depend on source-data quality, station identity matching, rainfall grid assignment, preprocessing and the selected holdout period.
- The reported model metrics are from one run and should be treated as preliminary comparative results, not universal performance guarantees.

## Data sources and references

1. Central Ground Water Board (CGWB), Ground Water Level Monitoring: https://cgwb.gov.in/en/ground-water-level-monitoring

2. India-WRIS groundwater information portal: https://indiawris.gov.in/wris/

3. India Meteorological Department (IMD), 0.25° daily gridded rainfall data: https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html

4. Pai, D. S., Sridhar, L., Rajeevan, M., Sreejith, O. P., Satbhai, N. S., & Mukhopadhyay, B. (2014). Development of a new high spatial resolution (0.25° × 0.25°) long period (1901–2010) daily gridded rainfall data set over India and its comparison with existing data sets over the region. *MAUSAM, 65*(1), 1–18.

5. Scikit-learn documentation, Regression metrics: https://scikit-learn.org/stable/modules/model_evaluation.html

## Author

Myra Chauhan
University Roll No.: 24020570017
B.Sc (Hons) Computer Science
Machine Learning (Semester-5)
Ramanujan College, Delhi University