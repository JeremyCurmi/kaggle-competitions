import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def diagonal_correlation(df: pd.DataFrame) -> (pd.DataFrame, np.ndarray):
    cm = df.corr()
    mask = np.zeros_like(cm, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    return cm, mask


def correlation_heatmap(df: pd.DataFrame):
    cm, mask = diagonal_correlation(df)
    fig = plt.figure(figsize=(8,8))
    sns.heatmap(cm, mask=mask, square=True, linewidths=0.1, annot=True, cbar=True)


def correlation_1d_plot(df: pd.DataFrame, feature: str):
    cm = df.corr()
    cm = cm.sort_values(by=feature)
    fig = plt.figure(figsize=(12, 4))
    fig = sns.scatterplot(data=cm, x=cm.index, y=feature)
    fig.axhline(0.7)
    fig.axhline(-0.7)
    _, labels = plt.xticks()
    fig = plt.setp(labels, rotation=45)


def grid_spec_scatter_plot(df: pd.DataFrame, feature1: str, feature2: str):
    # Create fig and gridspec
    fig = plt.figure(figsize=(16, 10), dpi=80)
    grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

    # Define the axes
    ax_main = fig.add_subplot(grid[:-1, :-1])
    ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
    ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

    ax_main.scatter(x=feature1, y=feature2, data=df, alpha=.4, cmap="coolwarm")

    # Boxplot on the right
    ax_right.boxplot(x=df[feature1])
    plt.xlabel(feature1)

    # boxplot on the bottom
    ax_bottom.boxplot(x=df[feature2], vert=False)

    # Decorations
    ax_main.set(title=f'Scatterplot with Boxplot \n {feature1} vs. {feature2}', ylabel=feature2)


def plot_feature_over_datetime(df: pd.DataFrame, date_feature: str, feature: str):
    if df[date_feature].dtype == object:
        df['date_time'] = pd.to_datetime(df[date_feature])
    plt.figure(figsize=(16,5))
    sns.lineplot(x=df['date_time'], y=df[feature])

