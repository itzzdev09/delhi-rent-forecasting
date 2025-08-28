# utils/metrics.py
"""
Evaluation metrics for Delhi House Rent Prediction Project.
"""

import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mse(y_true, y_pred):
    """Mean Squared Error"""
    return mean_squared_error(y_true, y_pred)


def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)


def r2(y_true, y_pred):
    """R-squared (coefficient of determination)"""
    return r2_score(y_true, y_pred)


def evaluate_regression(y_true, y_pred):
    """
    Compute all metrics at once.
    Returns dictionary.
    """
    return {
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }
