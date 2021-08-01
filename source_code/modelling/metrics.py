import pandas as pd
from sklearn import metrics
from typing import Dict


def calculate_regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    return {
        'r2': metrics.r2_score(y_true, y_pred),
        'mse': metrics.mean_squared_error(y_true, y_pred),
        'mae': metrics.mean_absolute_error(y_true, y_pred),
    }


def calculate_binary_classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    return {
        'accuracy': metrics.accuracy_score(y_true, y_pred),
        'precision': metrics.precision_score(y_true, y_pred),
        'recall': metrics.recall_score(y_true, y_pred),
        'f1': metrics.f1_score(y_true, y_pred),
    }


def calculate_multiclass_classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    return {
        'accuracy': metrics.accuracy_score(y_true, y_pred),
        'precision_micro': metrics.precision_score(y_true,y_pred, average='micro'),
        'recall_micro': metrics.recall_score(y_true, y_pred, average='micro'),
        'f1': metrics.f1_score(y_true, y_pred, average='micro'),
    }