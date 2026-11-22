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

