"""
Scalecast + Scikit-Learn Forecasting Pipeline
---------------------------------------------

A general-purpose time series forecasting pipeline using Scalecast and
Scikit-Learn regressors.

This template supports:
- Univariate and multivariate forecasting
- Feature engineering (seasonal/AR terms)
- Hyperparameter tuning
- Cross-validation
- Ensemble modeling (Bagging + Stacking)
- Confidence interval evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from scalecast.Forecaster import Forecaster
from scalecast.auxmodels import mlp_stack

# Optional ensemble models
from sklearn.ensemble import BaggingRegressor


def load_data(path: str) -> pd.DataFrame:
    """Load a time series dataset from CSV."""
    return pd.read_csv(path)


def initialize_forecaster(data: pd.DataFrame, target_column: str, date_column: str) -> Forecaster:
    """Initialize Forecaster with specified target and date columns."""
    return Forecaster(y=data[target_column], current_dates=data[date_column])


def perform_eda(f: Forecaster, title: str = 'Time Series Plot'):
    """Visualize the raw time series."""
    f.plot()
    plt.title(title, fontsize=16)
    plt.show()


def configure_forecaster(f: Forecaster, forecast_length: int = 30) -> Forecaster:
    """
    Configure forecast horizon, test/train split, confidence intervals,
    and add common seasonal/AR features.
    """
    f.generate_future_dates(forecast_length)
    f.set_test_length(.2)
    f.eval_cis()

    # Add time-based features (customize as needed)
    f.add_ar_terms(7)
    f.add_AR_terms((4, 7))
    f.add_seasonal_regressors('month', 'quarter', 'week', 'dayofyear', raw=False, sincos=True)
    f.add_seasonal_regressors('dayofweek', 'is_leap_year', raw=False, dummy=True, drop_first=True)
    f.add_seasonal_regressors('year')
    return f


def plot_test_results(f: Forecaster) -> pd.DataFrame:
    """Plot test-set results and return evaluation summary."""
    f.plot_test_set(models=f.estimator, ci=True)
    plt.title(f'{f.estimator} Test Set Results', fontsize=16)
    plt.show()

    return f.export('model_summaries', determine_best_by='TestSetMAPE')[
        ['ModelNickname', 'HyperParams', 'TestSetMAPE', 'TestSetR2', 'InSampleMAPE', 'InSampleR2']
    ]


def train_model(
    f: Forecaster,
    model_name: str,
    grid: dict = None,
    use_cv: bool = False,
    k: int = 3
) -> pd.DataFrame:
    """Train and evaluate a forecasting model with optional grid search and cross-validation."""
    f.set_estimator(model_name)
    if grid:
        f.ingest_grid(grid)
    if use_cv:
        f.cross_validate(k=k)
        f.auto_forecast()
    else:
        f.manual_forecast()
    return plot_test_results(f)


def train_custom_bagging(f: Forecaster) -> pd.DataFrame:
    """Train a BaggingRegressor using MLP as the base estimator."""
    f.set_estimator('bagging')
    f.manual_forecast(
        base_estimator=MLPRegressor(hidden_layer_sizes=(100, 100, 100), solver='lbfgs'),
        max_samples=0.9,
        max_features=0.5
    )
    return plot_test_results(f)


def train_stacking(f: Forecaster, estimators: list[str]) -> pd.DataFrame:
    """Train a Stacking model using provided base models."""
    f.set_estimator('stacking')
    mlp_stack(f, estimators, call_me='stacking')
    return plot_test_results(f)


def plot_forecast(f: Forecaster, model: str = 'stacking'):
    """Visualize the final forecast with confidence intervals."""
    f.plot(model, ci=True)
    plt.title(f'{model} Forecast', fontsize=16)
    plt.show()


# Example Grids for CV
lasso_grid = {'alpha': np.linspace(0, 2, 100)}
rf_grid = {
    'max_depth': [2, 3, 4, 5],
    'n_estimators': [100, 200],
    'max_features': ['auto', 'sqrt'],
    'max_samples': [0.75, 1]
}
xgboost_grid = {
    'n_estimators': [100, 200],
    'scale_pos_weight': [1, 5],
    'learning_rate': [0.1, 0.2],
    'gamma': [0, 3],
    'subsample': [0.8, 1.0]
}
lightgbm_grid = {
    'n_estimators': [100, 200],
    'boosting_type': ['gbdt', 'dart'],
    'max_depth': [2, 3],
    'learning_rate': [0.01, 0.1],
    'num_leaves': [10, 20],
}


def main():

    path = 'your_timeseries.csv'
    target_column = 'your_target_column'
    date_column = 'your_date_column'

    data = load_data(path)
    f = initialize_forecaster(data, target_column, date_column)
    perform_eda(f, title=target_column)
    f = configure_forecaster(f, forecast_length=60)


    print(train_model(f, 'mlr'))
    print(train_model(f, 'lasso', lasso_grid, use_cv=True))
    print(train_model(f, 'ridge', lasso_grid, use_cv=True))
    print(train_model(f, 'elasticnet', use_cv=True))
    print(train_model(f, 'rf', rf_grid, use_cv=True))
    print(train_model(f, 'xgboost', xgboost_grid, use_cv=True))
    print(train_model(f, 'lightgbm', lightgbm_grid, use_cv=True))
    print(train_model(f, 'sgd', use_cv=True))
    print(train_model(f, 'knn', use_cv=True))


    print(train_custom_bagging(f))
    base_models = ['knn', 'xgboost', 'lightgbm', 'sgd']
    print(train_stacking(f, base_models))


    plot_forecast(f, 'stacking')


if __name__ == '__main__':
    main()
