from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier
from feature_selection import feature_importance_from_tree_based_model
X, y = load_iris(return_X_y=True, as_frame=True)


def test_feature_importance_from_tree_based_model():
    expected = ['petal length (cm)', 'petal width (cm)']
    got, _ = feature_importance_from_tree_based_model(ExtraTreesClassifier(n_estimators=50), X, y, 2)
    assert got == expected


def test_feature_importance_from_tree_based_model_xgboost():
    expected = ['petal length (cm)', 'petal width (cm)']
    got, _ = feature_importance_from_tree_based_model(XGBClassifier(n_estimators=50), X, y, 2)
    assert got == expected
