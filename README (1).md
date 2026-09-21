# Advanced Time-Series Forecasting

## Overview

This project performs advanced time-series forecasting on household
electricity consumption data. It uses historical minute-level
measurements, converts them to hourly observations, creates time-series
and calendar-based features, performs exploratory data analysis, and
compares several machine-learning and deep-learning forecasting models.

The target variable is **`Global_active_power`**.

## Dataset

The project uses the **Individual Household Electric Power Consumption**
dataset.

-   Original frequency: minute-level measurements
-   Number of measurements: **2,075,259**
-   Period: **December 2006 to November 2010**
-   File format: semicolon-separated (`;`)
-   Missing values are represented by `?`
-   Date and time are combined into a single datetime index

The notebook expects the source file to be named:

``` text
household_power_consumption.txt
```

and to be available in the working directory.

## Project Workflow

The notebook follows these main steps:

1.  Import libraries
2.  Load and prepare the dataset
3.  Handle missing values
4.  Resample minute-level data to hourly data
5.  Perform feature engineering
6.  Explore the time series through EDA
7.  Split the data chronologically into training and testing sets
8.  Train traditional machine-learning models
9.  Train an LSTM neural network
10. Compare model performance
11. Visualize predictions and feature importance

## Technologies and Libraries

The project is implemented in Python and uses:

-   **NumPy** -- numerical operations
-   **Pandas** -- data loading, manipulation, resampling, and feature
    engineering
-   **Matplotlib** -- visualization
-   **Seaborn** -- correlation heatmap
-   **Scikit-learn** -- preprocessing, regression models, and evaluation
    metrics
-   **TensorFlow / Keras** -- LSTM deep-learning model

Random seeds are set to `42` for NumPy and TensorFlow to improve
reproducibility.

## Data Preprocessing

### Missing Values

Missing observations are handled using time-based interpolation:

-   Time interpolation with a limit of 60 observations
-   Forward fill (`ffill`)
-   Backward fill (`bfill`)

### Resampling

The original minute-level observations are resampled to **hourly
averages**.

Small gaps created during resampling are handled using time-based
interpolation, followed by removal of remaining missing rows.

## Feature Engineering

### Calendar Features

The following time-based features are created:

-   `hour`
-   `day_of_week`
-   `month`
-   `is_weekend`

### Cyclical Features

Because hours and months are periodic, cyclical encodings are created:

-   `hour_sin`
-   `hour_cos`
-   `month_sin`
-   `month_cos`

These allow models to represent the circular relationship between values
such as 23:00 and 00:00.

### Lag Features

Historical values of `Global_active_power` are included using:

-   1 hour
-   2 hours
-   3 hours
-   24 hours
-   48 hours
-   168 hours (1 week)

### Rolling Statistics

Rolling mean and standard deviation features are calculated for:

-   6 hours
-   24 hours
-   168 hours (1 week)

Rows containing missing values introduced by lag and rolling
calculations are removed.

## Exploratory Data Analysis

The notebook performs several EDA tasks:

-   Summary statistics
-   Full hourly time-series visualization
-   One-week zoomed time-series visualization
-   Average consumption by hour
-   Average consumption by day of week
-   Average consumption by month
-   Correlation matrix / heatmap for the main power-consumption
    variables

The EDA is intended to reveal temporal patterns, seasonality,
relationships between variables, and potential predictive signals.

## Train/Test Split

The dataset is split chronologically:

-   **80%** training data
-   **20%** test data

The split is performed according to time order rather than randomly,
which is appropriate for time-series forecasting.

For the traditional machine-learning models, `StandardScaler` is fitted
only on the training data and then applied to the test data to avoid
data leakage.

## Forecasting Models

### 1. Linear Regression

Linear Regression is used as the baseline model.

### 2. Random Forest Regressor

A Random Forest model is trained with:

-   `n_estimators=100`
-   `max_depth=15`
-   `random_state=42`
-   `n_jobs=-1`

It is used to capture nonlinear relationships between engineered
features and power consumption.

### 3. Gradient Boosting Regressor

The Gradient Boosting model uses:

-   `n_estimators=150`
-   `learning_rate=0.1`
-   `max_depth=5`
-   `random_state=42`

This model provides another nonlinear ensemble approach for forecasting.

### 4. LSTM

A Long Short-Term Memory neural network is used to model sequential
dependencies directly.

The LSTM uses the previous **24 hours** to predict the next hour.

Architecture:

``` text
Input: 24-hour sequence
        ↓
LSTM(64, return_sequences=True)
        ↓
Dropout(0.2)
        ↓
LSTM(32)
        ↓
Dropout(0.2)
        ↓
Dense(1)
```

Training configuration:

-   Optimizer: Adam
-   Loss: Mean Squared Error (`mse`)
-   Metric: Mean Absolute Error (`mae`)
-   Epochs: 20
-   Batch size: 64
-   Validation split: 10%
-   Early stopping patience: 5
-   Best validation weights are restored

The target series is scaled to `[0, 1]` using `MinMaxScaler` before
sequence construction and transformed back to the original scale for
evaluation.

## Evaluation Metrics

Models are evaluated using:

### MAE --- Mean Absolute Error

Measures the average absolute difference between predicted and actual
values.

### RMSE --- Root Mean Squared Error

Measures prediction error while giving greater weight to larger errors.

### R² --- R-squared

Measures how much of the variance in the target is explained by the
model.

The notebook creates a final comparison table containing:

``` text
Model
MAE
RMSE
R2
```

The models are sorted by RMSE in the final comparison.

> Note: The README does not report a specific winning model because the
> exact numerical results depend on running the notebook with the
> dataset.

## Visualization of Results

The notebook visualizes:

-   MAE, RMSE, and R² for all models
-   Actual vs. predicted values for the last 200 test hours
-   Random Forest feature importance
-   Top 15 features according to Random Forest importance

## Project Structure

A typical project directory can be organized as:

``` text
project/
│
├── Advanced Time - Series Forecasting .ipynb
├── household_power_consumption.txt
└── README.md
```

## Requirements

Install the required Python packages with:

``` bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

A Python environment with Jupyter Notebook or JupyterLab is also
recommended.

## How to Run

1.  Clone or download the project.
2.  Place `household_power_consumption.txt` in the same working
    directory as the notebook.
3.  Install the required dependencies.
4.  Open the notebook:

``` bash
jupyter notebook
```

5.  Open:

``` text
Advanced Time - Series Forecasting .ipynb
```

6.  Run the cells from top to bottom.

## Reproducibility

The notebook sets:

``` python
np.random.seed(42)
tf.random.set_seed(42)
```

to make NumPy and TensorFlow operations more reproducible.

## Notes

-   The notebook performs chronological splitting rather than random
    splitting because the data is time-dependent.
-   Feature scaling for the traditional ML models is fitted only on the
    training set.
-   The LSTM uses a 24-hour historical sequence to forecast the next
    hour.
-   The exact model metrics should be obtained by running the notebook;
    they are intentionally not hard-coded into this README.
-   The notebook assumes the original dataset file is available locally.

## Future Improvements

Possible extensions include:

-   Multi-step forecasting for several hours or days ahead
-   Hyperparameter tuning for all models
-   Walk-forward or rolling-window validation
-   More advanced sequence models such as GRU or Transformer
    architectures
-   Additional lag and rolling-window features
-   Separate models for different household consumption patterns
-   Saving trained models and scalers for production inference
-   Creating a forecasting API or dashboard

## License

No license information is specified in the notebook. Add an appropriate
license if this project is intended for public distribution.
