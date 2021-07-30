import pandas as pd
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import chi2
from sklearn.datasets import load_iris, load_digits
from source_code.feature_selection import select_features_from_model, univariate_feature_selection, rfe_feature_selection

X, y = load_iris(return_X_y=True, as_frame=True)


def test_select_features_from_model():
    expected = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']
    got = select_features_from_model(LinearSVC(C=0.01, penalty="l1", dual=False), X, y)
    assert got == expected


def test_univariate_feature_selection():
    expected = ['petal length (cm)', 'petal width (cm)']
    got = univariate_feature_selection(X, y, 2, chi2)
    assert got == expected


def test_rfe_feature_selection():
    expected = [30, 38, 42, 46, 53]
    digits = load_digits()
    X = pd.DataFrame(digits.images.reshape((len(digits.images), -1)))
    y = pd.Series(digits.target)
    model = SVC(kernel="linear", C=1)
    got = rfe_feature_selection(model, X, y, 5)
    assert got == expected
