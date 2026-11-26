# Groundwater-Level Forecasting and Trend Analysis

A machine-learning project for analysing historical groundwater-level behaviour and evaluating regression models for groundwater-depth prediction in **Karnal and Kurukshetra, Haryana, India**.

> **Interpretation:** Groundwater level is recorded as depth below ground level (mbgl). A larger depth value means the water table is farther below the ground surface. Therefore, a positive long-term slope indicates a declining groundwater-level trend.

## Project Objectives

- Prepare seasonal groundwater observations for Karnal and Kurukshetra.
- Combine groundwater observations with antecedent rainfall and previous groundwater measurements.
- Train and compare five regression models.
- Evaluate predictions using MAE, RMSE and R² on a chronological holdout test set.
- Apply chronological cross-validation to compare model performance and support model selection.
- Independently analyse station-wise pre-monsoon and post-monsoon groundwater trends and data sufficiency.
- Perform sensitivity analysis to investigate the effect of an anomalous groundwater observation.
- Visualise groundwater trends, monitoring-data reliability, station histories and model performance.

## Study Area and Data

The study focuses on two districts in Haryana:

- **Karnal**
- **Kurukshetra**

The groundwater source is the Central Ground Water Board (CGWB) / India-WRIS groundwater monitoring data. Rainfall features were derived from the India Meteorological Department (IMD) daily gridded rainfall product at 0.25° spatial resolution. For each monitoring-well location, the nearest grid cell was used, and daily rainfall was aggregated into monthly totals.

The processed seasonal modelling dataset contains **1,882 observations**:

| District | Total observations | Pre-monsoon | Post-monsoon |
|---|---:|---:|---:|
| Karnal | 1,282 | 577 | 705 |
| Kurukshetra | 600 | 298 | 302 |
| **Total** | **1,882** | **875** | **1,007** |

The modelling dataset covers **15 May 1997 to 1 November 2023**. The selected target observations are May (pre-monsoon) and November (post-monsoon).

The source records are irregularly spaced. Groundwater lag features represent previous available observations for the same monitoring well and should not be interpreted as guaranteed consecutive calendar quarters.

### Modelling Variables

**Target variable:**

`Groundwater Level Quarterly Manual (meter)`

**Predictor variables:**

- `GWL_Lag_1`
- `GWL_Lag_2`
- `GWL_Lag_3`
- `Rainfall_prev_1m`
- `Rainfall_prev_3m`
- `Rainfall_prev_6m`

Groundwater lags are calculated within each unique well, identified using district, station name, latitude and longitude. Rainfall is matched to the well location using the nearest IMD grid cell.

## Methodology

### 1. Data Preprocessing

The preprocessing stage involves:

1. Parsing observation dates and sorting measurements chronologically within each well.
2. Retaining the study districts and selected seasonal observations.
3. Deriving monthly rainfall totals and antecedent rainfall windows.
4. Constructing three previous-observation groundwater lag features per well.
5. Removing records that lack required predictor or target values.
6. Preparing the final modelling dataset for regression experiments.

### 2. Training and Testing Strategy

A chronological holdout strategy was used to evaluate model performance on later observations.

- Training observations: 1,475
- Testing observations: 407
- Training period ends: 30 November 2020
- Testing period begins: 30 November 2021

The split maintains temporal ordering and has no overlapping observation dates between the training and testing partitions. A time gap occurs because the source observations are irregularly spaced.

### 3. Regression Models

Five regression algorithms were implemented and compared.

**Linear Regression with Lags**

Linear Regression establishes a baseline relationship between groundwater levels and the selected predictor variables. It estimates coefficients by minimising the sum of squared prediction errors.

**Polynomial Regression (Degree 2)**

Polynomial Regression extends the linear model by introducing squared and interaction terms. This allows the model to capture certain nonlinear relationships between the predictors and groundwater levels.

**Ridge Regression**

Ridge Regression is a regularised form of Linear Regression that adds an L2 penalty to the loss function. This penalises excessively large coefficients and helps reduce overfitting, particularly when predictors are correlated.

The Ridge model uses `alpha = 10` in the reported experiment. This configuration was selected through chronological cross-validation.

**K-Nearest Neighbours (KNN) Regressor**

KNN predicts groundwater levels using the average target values of nearby observations in feature space. Feature scaling is important because the algorithm relies on distance calculations.

**Multi-Layer Perceptron (MLP) Regressor**

MLP is a feed-forward artificial neural network capable of learning nonlinear relationships. The reported configuration uses two hidden layers containing 100 and 50 units, respectively, with a maximum of 2,000 iterations and a fixed random seed.

Feature scaling is applied within the model pipelines.

### 4. Chronological Cross-Validation

Three expanding-window chronological cross-validation folds were used to compare model performance.

The mean cross-validation RMSE values were:

| Model | Mean CV RMSE (m) |
|---|---:|
| Linear Regression | 3.128 |
| Polynomial Regression (Degree 2) | 3.367 |
| Ridge Regression (alpha = 10) | 3.103 |
| KNN Regressor | 3.349 |
| MLP Regressor | 3.115 |

Ridge Regression achieved the lowest average cross-validation RMSE and was therefore selected as the preferred model based on cross-validation.

The final holdout test results are reported separately because the model with the best cross-validation score does not necessarily achieve the best performance on the final test set.

### 5. Evaluation Metrics

The models were evaluated using three metrics:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted groundwater levels. Lower values indicate better performance.
- **RMSE (Root Mean Squared Error):** Measures the square root of the average squared prediction error. Lower values indicate better performance, with larger errors receiving more weight.
- **R² (Coefficient of Determination):** Measures how much of the variation in the target variable is explained by the model. Higher values generally indicate better performance. R² is not a percentage accuracy measure.

## Model Results

The following table presents the final chronological holdout test results from the reviewer experiments.

| Model | MAE (m) | RMSE (m) | R² |
|---|---:|---:|---:|
| Linear Regression with Lags | 2.208 | 3.210 | 0.9222 |
| Polynomial Regression (Degree 2) | 2.095 | 3.317 | 0.9170 |
| Ridge Regression (alpha = 10) | 2.196 | 3.218 | 0.9219 |
| KNN Regressor | 2.384 | 3.435 | 0.9110 |
| MLP Regressor | 1.937 | 2.973 | 0.9333 |

### Result Interpretation

The MLP Regressor achieved the lowest MAE (1.937 m), lowest RMSE (2.973 m) and highest R² (0.9333) on the final holdout test set.

Ridge Regression achieved the best average RMSE during chronological cross-validation (3.103 m), demonstrating strong validation performance.

These findings distinguish two different evaluation outcomes:

- **Cross-validation selection:** Ridge Regression.
- **Best final holdout performance:** MLP Regressor.

The MLP's stronger holdout performance does not establish that it will perform best on other districts, time periods or previously unseen monitoring wells.

## Historical Station-Wise Trend Analysis

Historical trend analysis is a separate component from the machine-learning prediction experiment. It uses station-season observations to fit a linear trend against time.

### Seasonal Definitions

- **Pre-monsoon:** May.
- **Post-monsoon:** January and November.

### Data Reliability Classification

| Classification | Number of observations |
|---|---:|
| Reliable | At least 10 |
| Limited | 5–9 |
| Insufficient | Fewer than 5 |

These reliability thresholds are project-defined and are not official CGWB classifications.

### Trend Classification

The station-wise trend is calculated using linear regression, with the slope expressed in metres per year.

A threshold of ±0.05 m/year is used:

- **Declining:** Slope greater than +0.05 m/year.
- **Improving:** Slope lower than -0.05 m/year.
- **Stable:** Slope between -0.05 and +0.05 m/year.

A positive slope indicates increasing groundwater depth and therefore a declining water table. A negative slope indicates decreasing groundwater depth and therefore an improving water table.

Station records are distinguished using district, station name, latitude and longitude.

Trend classifications are exploratory screening results and should not be interpreted as official CGWB groundwater-resource categories.

## Sensitivity Analysis

A sensitivity analysis was conducted to investigate the effect of an anomalous **204.5 m lagged groundwater-level observation** associated with Shahbad 1(d), Kurukshetra.

The analysis:

- Identifies the observation containing the 204.5 m lag value.
- Checks its occurrence in the training, cross-validation and testing partitions.
- Compares model performance using the complete chronological holdout set.
- Repeats the evaluation after excluding the affected test observation.
- Retains the original observation in the primary analysis rather than silently removing it.

The sensitivity experiment compares seven models, including the persistence and training-mean baselines, alongside the five regression models.

The results show that the anomalous observation substantially affects the persistence baseline. The MLP retains the lowest RMSE and highest R² after excluding the affected observation, while the persistence baseline achieves the lowest MAE in the reduced test set.

To reproduce the analysis, run:

```bash
python src/reviewer_experiments.py
python src/sensitivity_analysis_204_5m.py
```

Results are saved in:

`results/reviewer_revision/`

## Repository Structure

```text
.
├── data/
│   ├── groundwater_final_dataset.csv
│   └── gwl_manual_quarterly_cgwb_hr_1991_2025.csv
│
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── model_training.py
│   ├── trend_analysis.py
│   ├── reviewer_experiments.py
│   └── sensitivity_analysis_204_5m.py
│
├── results/
│   ├── model_comparison.csv
│   ├── groundwater_trend_results.csv
│   ├── graphs/
│   │   ├── model_evaluation/
│   │   └── trend_analysis/
│   └── reviewer_revision/
│       ├── graphs/
│       ├── predictions/
│       └── tables/
│
├── main.py
├── requirements.txt
└── ReadMe.md
```

The repository includes the raw groundwater dataset used in the project and the processed modelling dataset. Rainfall-derived features are included in the processed dataset.

## Running the Project Locally

### Requirements

- Python 3.10 or later
- Git
- Required Python libraries listed in `requirements.txt`

### Installation

Clone the repository:

```bash
git clone https://github.com/myrayaya/groundwater-level-forecasting-ml.git

cd groundwater-level-forecasting-ml
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Execute the Project

Run the scripts from the project root directory:

```bash
python main.py
python src/preprocessing.py
python src/trend_analysis.py
python src/model_training.py
python src/reviewer_experiments.py
python src/sensitivity_analysis_204_5m.py
```

Run the sensitivity analysis after the reviewer experiments.

The scripts generate the corresponding datasets, evaluation tables and visualisations inside the results directory.

## Limitations

- Groundwater monitoring observations are irregularly spaced. A previous-observation lag is not necessarily a one-quarter lag.
- The chronological split evaluates later observations from the overall set of monitoring wells. It is not a fully independent test on unseen wells.
- The global time split leaves a gap between the latest training date and earliest test date because observations are irregular.
- The test set was inspected during development, so the final holdout results may be somewhat optimistic.
- The trend slope is a simple linear summary and does not model all hydrogeological, pumping, land-use or recharge influences.
- Rainfall is only one influence on groundwater levels. The models do not establish causal effects.
- The rainfall features did not improve performance in the fixed Ridge feature-ablation experiment. This does not establish that rainfall is generally unimportant.
- Results depend on source-data quality, station identity matching, rainfall grid assignment, preprocessing and the selected holdout period.
- The reported model metrics are from one experimental run and should be treated as comparative results, not universal performance guarantees.
- Generalisation to completely unseen wells and other geographical regions has not been independently established.

## Data Sources and References

1. Central Ground Water Board (CGWB), Ground Water Level Monitoring:  
   https://cgwb.gov.in/en/ground-water-level-monitoring

2. India-WRIS Groundwater Information Portal:  
   https://indiawris.gov.in/wris/

3. India Meteorological Department (IMD), 0.25° Daily Gridded Rainfall Data:  
   https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html

4. Pai, D. S., Sridhar, L., Rajeevan, M., Sreejith, O. P., Satbhai, N. S., & Mukhopadhyay, B. (2014). Development of a new high spatial resolution (0.25° × 0.25°) long period (1901–2010) daily gridded rainfall data set over India and its comparison with existing data sets over the region. *MAUSAM, 65*(1), 1–18.

5. Scikit-learn Documentation, Model Evaluation and Regression Metrics:  
   https://scikit-learn.org/stable/modules/model_evaluation.html

## Author

**Myra Chauhan**  
B.Sc. (Hons.) Computer Science  
Machine Learning, Semester V  
Ramanujan College, University of Delhi