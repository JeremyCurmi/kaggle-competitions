from sklearn.linear_model import LinearRegression, Lasso, BayesianRidge, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC, LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from tqdm import tqdm
import pandas as pd
from sklearn import model_selection

REGRESSION_MODELS = [
    ('lr',LinearRegression()),
    ('lasso',Lasso()),
    ('baye_ridge',BayesianRidge()),
    ('mlp',MLPRegressor()),
    ('knn',KNeighborsRegressor()),
    ('svr',SVR()),
    ('linear_svr',LinearSVR()),
    ('rf',RandomForestRegressor()),
    ('xgb',XGBRegressor()),
    ('lgbm',LGBMRegressor()),
]

CLASSIFICATION_MODELS = [
    ('lr',LogisticRegression()),
    ('mlp',MLPClassifier()),
    ('knn',KNeighborsClassifier()),
    ('svr',SVC()),
    ('linear_svr',LinearSVC()),
    ('rf',RandomForestClassifier()),
    ('xgb',XGBClassifier()),
    ('lgbm',LGBMClassifier()),
]

REGRESSION_SCORES = ['explained_variance','r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
CLASSIFICATION_SCORES = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']


def run_experiments(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, is_classification: bool = True):
    dfs = []

    if is_classification:
        models = CLASSIFICATION_MODELS.copy()
        scores = CLASSIFICATION_SCORES.copy()
    else:
        models = REGRESSION_MODELS.copy()
        scores = REGRESSION_SCORES.copy()

    # TODO Convert to multiprocessing
    # https://johaupt.github.io/python/parallel%20processing/cross-validation/multiprocessing_cross_validation.html

    for name, model in tqdm(models):
        kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=123)
        cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scores)

        ml = model.fit(X_train, y_train)
        y_pred = ml.predict(X_test)

        # TODO calculate metrics on validation set
        ...

        df_model_performance = pd.DataFrame(cv_results)
        df_model_performance['model'] = name
        dfs.append(df_model_performance)

    return pd.concat(dfs,ignore_index=True)

# TODO evaluate results from run_experiments function
# https://towardsdatascience.com/quickly-test-multiple-models-a98477476f0