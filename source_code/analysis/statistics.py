import numpy as np
import pandas as pd
import seaborn as sns
from typing import List
from regressors import stats
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy.stats import kurtosis, skew, probplot
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from statsmodels.stats.outliers_influence import variance_inflation_factor


def skew_and_kurtosis_table(df: pd.DataFrame):
    # TODO DELETE THIS FUNCTION AND CREATE A NEW FUNCTION WHIHC COMPUTES STATS FOR EACH FEATURE AND COMPARE BETWEEN TRAIN AND TEST DATA

    sk_dict = {"Skewness": skew(df),
               "Kurtosis": kurtosis(df)}

    return pd.DataFrame(sk_dict, index=df.columns).style.background_gradient(subset=["Skewness", "Kurtosis"])


def ols_summary_statistics(X: pd.DataFrame, y: pd.Series) -> (LinearRegression, pd.DataFrame):
    ols = LinearRegression().fit(X, y)
    stats.summary(ols, X, y, X.columns.tolist())
    return ols, ols_summary_df(ols, X, y)


def ols_summary_df(lr: LinearRegression, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    stats_parameters = [lr, X, y]
    features = ['intercept']
    features.extend(X.columns)
    return pd.DataFrame({
        "estimates": np.append(lr.intercept_, lr.coef_),
        "std-error": stats.coef_se(*stats_parameters),
        "t-value": stats.coef_tval(*stats_parameters),
        "p-value": stats.coef_pval(*stats_parameters)
    }, index=features)


def plot_residuals(model, X: pd.DataFrame, y: pd.Series) -> pd.Series:
    plt.figure(figsize=(16, 7))
    residuals = pd.Series(y - model.predict(X), name="residuals")

    # Plot scatterplot
    sns.scatterplot(x=y, y=residuals)
    plt.axhline(y=0, color='red', linestyle='--')
    return residuals


def plot_residuals_dist(model, X: pd.DataFrame, y: pd.Series):
    plt.figure(figsize=(10,6))
    residuals = pd.Series(y - model.predict(X), name='residuals')
    sns.histplot(x=residuals)
    plt.title(f"{y.name} residuals distribution")


def probability_plot(model, X: pd.DataFrame, y: pd.Series):
    plt.figure(figsize=(10,6))
    residuals = pd.Series(y-model.predict(X), name='residuals')
    ax = plt.subplot(1,1,1)
    plt.title(f"{y.name} Probability Plot")
    return probplot(residuals, plot=ax)


def vif_multicollinearity_analysis(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str]):
    vif = pd.DataFrame()

    # Normalize data first
    sc = StandardScaler()
    scaled_train_df = sc.fit_transform(train_df[features])
    scaled_test_df = sc.transform(test_df[features])

    vif['features'] = features
    vif['vif_train'] = [variance_inflation_factor(scaled_train_df, i) for i in range(train_df[features].shape[1])]
    vif['vif_test'] = [variance_inflation_factor(scaled_test_df, i) for i in range(test_df[features].shape[1])]
    vif.set_index('features', inplace=True)

    return vif.style.background_gradient()


def dicky_fuller_test_for_stationarity(feature: pd.Series) -> pd.Series:
    df_test = adfuller(feature, autolag='AIC')
    df_test_output = pd.Series(df_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key, value in df_test[4].items():
        df_test_output['Critical Value (%s)'%key] = value
    return df_test_output
