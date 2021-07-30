import logging
import pandas as pd
from tqdm import tqdm
from sklearn import model_selection, metrics

from sklearn.linear_model import LinearRegression, Lasso, BayesianRidge, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC, LinearSVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

from source_code.modelling import calculate_regression_metrics, calculate_binary_classification_metrics, calculate_multiclass_classification_metrics

REGRESSION_MODELS = [
    ('lr',LinearRegression()),
    ('lasso',Lasso()),
    ('baye_ridge',BayesianRidge()),
    ('mlp',MLPRegressor()),
    ('knn',KNeighborsRegressor()),
    ('svm',SVR()),
    ('linear_svm',LinearSVR()),
    ('rf',RandomForestRegressor()),
    ('xgb',XGBRegressor()),
    ('lgbm',LGBMRegressor()),
]

CLASSIFICATION_MODELS = [
    ('lr',LogisticRegression()),
    ('mlp',MLPClassifier()),
    ('knn',KNeighborsClassifier()),
    ('svm',SVC(probability=True)),
    ('rf',RandomForestClassifier()),
    ('xgb',XGBClassifier()),
    ('lgbm',LGBMClassifier()),
]

REGRESSION_SCORES = ['explained_variance', 'r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
BINARY_CLASSIFICATION_SCORES = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'neg_log_loss']
MULTICLASS_CLASSIFICATION_SCORES = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro']


def run_experiments(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, is_classification: bool = True):
    dfs = []

    if is_classification:
        if len(y_train.value_counts()) == 2:
            models = CLASSIFICATION_MODELS.copy()
            scores = BINARY_CLASSIFICATION_SCORES.copy()
            get_metrics = calculate_binary_classification_metrics
        else:
            models = CLASSIFICATION_MODELS.copy()
            scores = MULTICLASS_CLASSIFICATION_SCORES.copy()
            get_metrics = calculate_multiclass_classification_metrics
    else:
        models = REGRESSION_MODELS.copy()
        scores = REGRESSION_SCORES.copy()
        get_metrics = calculate_regression_metrics

    # TODO Convert to multiprocessing
    # https://johaupt.github.io/python/parallel%20processing/cross-validation/multiprocessing_cross_validation.html

    for name, model in tqdm(models):
        k_fold = model_selection.KFold(n_splits=5, shuffle=True, random_state=123)
        cv_results = model_selection.cross_validate(model, X_train, y_train, cv=k_fold, scoring=scores)

        ml = model.fit(X_train, y_train)
        y_pred = ml.predict(X_test)

        validation_performance = get_metrics(y_test, y_pred)

        logging.info('renaming validation keys ...')
        keys = validation_performance.keys()
        for key in list(keys):
            validation_performance['validation_'+key] = validation_performance.pop(key)

        model_performance = {**cv_results, **validation_performance}
        df_model_performance = pd.DataFrame(model_performance).abs()
        df_model_performance['model'] = name
        dfs.append(df_model_performance)

    return pd.concat(dfs, ignore_index=True)

# TODO evaluate results from run_experiments function
# https://towardsdatascience.com/quickly-test-multiple-models-a98477476f0