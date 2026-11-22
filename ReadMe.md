# Groundwater-Level Forecasting and Trend Analysis

A machine-learning project for analysing historical groundwater-level behaviour and evaluating regression models for groundwater-depth prediction in Karnal and Kurukshetra, Haryana, India.

> **Interpretation:** Groundwater level is recorded as depth below ground level (mbgl). A larger depth value means the water table is farther below the ground surface. A positive long-term slope therefore indicates a declining groundwater-level trend.

## Project Objectives

- Prepare seasonal groundwater observations for Karnal and Kurukshetra.
- Combine groundwater observations with antecedent rainfall and previous groundwater measurements.
- Train and compare five regression models.
- Evaluate predictions using MAE, RMSE, R², on a chronological holdout set.
- Independently analyse station-wise pre-monsoon and post-monsoon groundwater trends and data sufficiency.
- Visualize district-level trends, monitoring-data reliability, selected station histories and model performance.

## Study area and data

The study focuses on two districts in Haryana: Karnal and Kurukshetra.

The groundwater source is the Central Ground Water Board (CGWB) / India-WRIS groundwater monitoring data. The rainfall features were derived from the India Meteorological Department (IMD) daily gridded rainfall product at 0.25° spatial solution. For each monitoring-well location, the nearest grid cell was used and daily rainfall was aggregated to monthly totals.

The processed seasonal modelling dataset contains **1,882 observations** :

- Karnal: 1,282 observations (577 pre-monsoon, 705 post-monsoon)
- Kurukshetra: 600 observations (298 pre-monsoon, 302 post-monsoon)

The modelling dataset covers 15 May 1997 to 1 November 2023. The selected target observations are May (pre-monsoon) and November (post-monsoon). The source records are irregularly spaced; lag features represent previous available observations for the same monitoring well and should not be interpreted as guaranteed consecutive calendar quarters.

## Modelling Variables

**Target:** `Groundwater Level Quarterly Manual (Meter)`

**Predictors:**

- `GWL_Lag_1` , `GWL_Lag_2` , `GWL_Lag_3`
- `Rainfall_prev_1m` , `Rainfall_prev_3m` , `Rainfall_prev_6m`

The groundwater lags are calculated within each unique well, identified using district, station name, latitude and longitude. Rainfall is matched to the well location using the nearest IMD grid cell.

### Methodology

#### Preprocessing

1. Parse observation dates and sort measurements chronologically within each well.
2. Retain the study districts and selected seasonal observations.
3. Derive monthly rainfall totals and antecedent rainfall windows.
4. Construct three previous-observation groundwater lag features per well.
5. Remove records that lack required predictor or target values.
6. Use a chronological 80/20 holdout split, keeping observations at the cutoff timestamp together.

#### Regression Models

1. Linear Regression with Lags
2. Polynomial Regression (degree 2)
3. Ridge Regression
4. K-Nearest Neighbours (KNN) Regressor
5. Multi-Layer Perceptron (MLP) Regressor

Feature scaling is used in the model pipelines. The MLP configuration used in the reported run has hidden layers of 100 and 50 units, with a maximum of 2,000 iterations and a fixed random seed.

#### Evaluation Metrics

- **MAE:** average absolute prediction error; lower is better.
- **RMSE:** square root of mean squared error; lower is better and larger errors receive more weight.
- **R²:** coefficient of determination; higher is generally better. R² is not a percentage accuracy measure.

