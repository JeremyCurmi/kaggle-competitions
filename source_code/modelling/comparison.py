import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection, metrics

from sklearn.linear_model import LinearRegression, Lasso, BayesianRidge, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC, LinearSVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

from source_code.modelling import calculate_regression_metrics, calculate_binary_classification_metrics, \
    calculate_multiclass_classification_metrics

REGRESSION_MODELS = [
    ('lr', LinearRegression),
    ('lasso', Lasso),
    ('baye_ridge', BayesianRidge),
    ('mlp', MLPRegressor),
    ('knn', KNeighborsRegressor),
    ('svr', SVR),
    ('linear_svm', LinearSVR),
    ('rf', RandomForestRegressor),
    ('xgb', XGBRegressor),
    ('lgbm', LGBMRegressor),
]

CLASSIFICATION_MODELS = [
    ('lr', LogisticRegression),
    ('mlp', MLPClassifier),
    ('knn', KNeighborsClassifier),
    ('svc', SVC),
    ('rf', RandomForestClassifier),
    ('xgb', XGBClassifier),
    ('lgbm', LGBMClassifier),
]

REGRESSION_SCORES = ['explained_variance', 'r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
BINARY_CLASSIFICATION_SCORES = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'neg_log_loss']
MULTICLASS_CLASSIFICATION_SCORES = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro']


def run_experiments(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
                    is_classification: bool = True):
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
        model = model(probability=True) if name == 'svc' else model()

        k_fold = model_selection.KFold(n_splits=5, shuffle=True, random_state=123)
        cv_results = model_selection.cross_validate(model, X_train, y_train, cv=k_fold, scoring=scores)

        ml = model.fit(X_train, y_train)
        y_pred = ml.predict(X_test)

        validation_performance = get_metrics(y_test, y_pred)

        logging.info('renaming validation keys ...')
        keys = validation_performance.keys()
        for key in list(keys):
            validation_performance['validation_' + key] = validation_performance.pop(key)

        model_performance = {**cv_results, **validation_performance}
        df_model_performance = pd.DataFrame(model_performance).abs()
        df_model_performance['model'] = name
        dfs.append(df_model_performance)

    return pd.concat(dfs, ignore_index=True)


def analyse_model_experiments(results: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    bootstraps = []

    for model in results['model'].unique():
        model_df = results.loc[results['model'] == model]

        bootstrap = model_df.sample(n=30, replace=True)
        bootstraps.append(bootstrap)

    bootstrap_df = pd.concat(bootstraps, ignore_index=True)
    results_vert_df = pd.melt(bootstrap_df, id_vars=['model'], var_name='metric', value_name='value')

    metrics_ = results_vert_df['metric'].unique()
    time_metrics = ['fit_time', 'score_time']
    validation_metrics = [x for x in metrics_ if 'validation_' in x]
    cv_metrics = [x for x in metrics_ if x not in time_metrics + validation_metrics]

    # fit time metrics
    time_results = results_vert_df.loc[results_vert_df['metric'].isin(time_metrics)]
    time_results = time_results.sort_values(by='value')

    # validation metrics
    validation_results = results_vert_df.loc[results_vert_df['metric'].isin(validation_metrics)]
    validation_results = validation_results.sort_values(by='value')

    # Performance metrics
    cv_results = results_vert_df.loc[results_vert_df['metric'].isin(cv_metrics)]
    cv_results = cv_results.sort_values(by='model')

    metrics_df = bootstrap_df.groupby(['model'])[metrics_].agg([np.std, np.mean])

    return metrics_df, time_results, validation_results, cv_results


def visualise_model_experiments(results: pd.DataFrame):
    metrics_df, time_results, validation_results, cv_results = analyse_model_experiments(results)

    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    # fig.title('Baseline Model Comparison')
    sns.set(font_scale=1.5)
    g1 = sns.boxplot(x="model", y="value", hue="metric", data=time_results, palette="Set3", ax=axes[0,0])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    axes[0,0].set_title('Comparison of Model by Fit and Score time')

    g2 = sns.barplot(x="model", y="value", hue="metric", data=validation_results, palette="Set3", ax=axes[0,1])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    axes[0,1].set_title('Comparison of Model by Validation results')

    cv_metrics = cv_results['metric'].unique()
    g3 = sns.boxplot(x="model", y="value", hue="metric", data=cv_results.loc[cv_results['metric'].isin(cv_metrics[:3])], palette="Set3", ax=axes[1,0])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    axes[1,0].set_title('Comparison of Model by CV results')

    g4 = sns.boxplot(x="model", y="value", hue="metric", data=cv_results.loc[~cv_results['metric'].isin(cv_metrics[:3])],palette="Set3", ax=axes[1,1])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    axes[1, 1].set_title('Comparison of Model by CV results Contd')

    fig.tight_layout()
    return metrics_df, fig
