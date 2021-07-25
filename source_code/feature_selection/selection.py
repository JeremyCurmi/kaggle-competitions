import pandas as pd
from typing import List, Callable
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFE


def select_features_from_model(model: BaseEstimator, X: pd.DataFrame, y: pd.Series, threshold: float = None) \
        -> List[str]:
    """


    :param model: unfitted model
    :param X: Feature space used to train the model
    :param y: target feature
    :param threshold: A threshold value to select features which have higher importance than this set value
    :return: list of features
    """
    selector = SelectFromModel(model, threshold=threshold).fit(X, y)

    feature_idx = selector.get_support()
    return X.columns[feature_idx].tolist()


def univariate_feature_selection(X: pd.DataFrame, y: pd.Series, n: int, score_func: Callable) -> List[str]:
    """

    :param X: Feature Space which has been scaled, preferably over [0,1] ... Note: for some score_func X > 0
    :param y: target feature
    :param n: number of features to return
    :param score_func: statistical test used for feature selection
    :return: list of n most correlated features with target feature
    """
    selector = SelectKBest(score_func=score_func, k=n).fit(X, y)
    feature_idx = selector.get_support()
    return X.columns[feature_idx].tolist()


def rfe_feature_selection(model: BaseEstimator, X: pd.DataFrame, y: pd.Series, n: int = None) -> List[str]:
    """

    :param model: unfitted model
    :param X:  Feature Space, preferably scaled
    :param y: target feature
    :param n: number of features to return (optional)
    :return: list of n most important features to the model
    """
    rfe = RFE(model, n).fit(X, y)
    feature_idx = rfe.get_support()
    return X.columns[feature_idx].tolist()
