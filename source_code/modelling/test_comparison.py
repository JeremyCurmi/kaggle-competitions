import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer

from source_code.modelling import run_experiments, analyse_model_experiments, visualise_model_experiments

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def test_run_experiments():
    # Binary Classification
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    got = run_experiments(X_train, X_test, y_train, y_test, True)

    # Multiclass Classification:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    got = run_experiments(X_train, X_test, y_train, y_test, True)

    # Regression
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    got = run_experiments(X_train, X_test, y_train, y_test, False)
    assert True


def test_analyse_model_experiments():
    # Binary Classification
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    _, _, _, _ = analyse_model_experiments(run_experiments(X_train, X_test, y_train, y_test, True))
    assert True


def test_visualise_model_experiments():
    # Binary Classification
    import matplotlib.pyplot as plt
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    metrics_df, _ = visualise_model_experiments(run_experiments(X_train, X_test, y_train, y_test, True))
    print(metrics_df)
    plt.show()
    assert True
