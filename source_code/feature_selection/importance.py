import pandas as pd
from typing import List
from sklearn.base import BaseEstimator


def feature_importance_from_tree_based_model(model: BaseEstimator, X: pd.DataFrame, y: pd.Series, n: int = None) \
        -> (List[str], List[float]):
    """

    :param model: tree based model used to compute feature importance
    :param X: Feature Space
    :param y: target label
    :param n: number of features to return (by importance)
    :return: feature importance features and scores
    """
    feat_imp = pd.Series(data=model.fit(X, y).feature_importances_, index=X.columns)

    if n:
        feat_imp = feat_imp.nlargest(n)

    return feat_imp.index.tolist(), list(feat_imp.values)
