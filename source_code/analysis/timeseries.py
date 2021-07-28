import logging
import pandas as pd
import seaborn as sns
from typing import List
import matplotlib.pyplot as plt


def timeseries_plot(train_df: pd.DataFrame, test_df: pd.DataFrame, date_feature: str, feature_list: List[str]):
    """
    Plot was copied from TPS July 2021 EDA created by Sharlto Cope.
    """
    rows = len(feature_list)
    plt.rcParams['figure.dpi'] = 600
    fig = plt.figure(figsize=(10, 8), facecolor='#f6f5f5')
    gs = fig.add_gridspec(rows, 1)
    gs.update(wspace=0, hspace=1.5)

    background_color = "#f6f5f5"

    run_no = 0
    logging.info("Creating figure ...")
    for row in range(0, rows):
        for col in range(0, 1):
            locals()["ax" + str(run_no)] = fig.add_subplot(gs[row, col])
            locals()["ax" + str(run_no)].set_facecolor(background_color)
            for s in ["top", "right"]:
                locals()["ax" + str(run_no)].spines[s].set_visible(False)
            run_no += 1

    run_no = 0
    logging.info("Plotting feature plots ...")
    for col in feature_list:
        sns.lineplot(ax=locals()["ax" + str(run_no)], y=train_df[col], x=pd.to_datetime(train_df[date_feature]),
                     color='#fcd12a')
        sns.lineplot(ax=locals()["ax" + str(run_no)], y=test_df[col], x=pd.to_datetime(test_df[date_feature]),
                     color='#287094')
        locals()["ax" + str(run_no)].set_ylabel('')
        locals()["ax" + str(run_no)].set_xlabel(col, fontsize=5, fontweight='bold')
        locals()["ax" + str(run_no)].tick_params(labelsize=5, width=0.5, length=1.5)
        locals()["ax" + str(run_no)].grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.7)
        locals()["ax" + str(run_no)].grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.7)
        run_no += 1

    plt.title('Showing time series data starting from train dataset followed by test dataset', fontsize=5)
    fig.legend(['test', 'train'], ncol=2, facecolor=background_color, edgecolor=background_color, fontsize=4,
               bbox_to_anchor=(0.2, 0.895))


def rolling_statistics(df: pd.DataFrame, features: List[str], date_feature: str):
    for i, col in enumerate(features):
        roll_mean = df[col].rolling(window=24).mean()
        roll_std = df[col].rolling(window=24).std()

        plt.figure(figsize=(25, 15))
        plt.subplot(len(features), 1, i + 1)
        sns.lineplot(x=date_feature, y=col, data=df, label=col)
        sns.lineplot(x=date_feature, y=roll_mean, data=df, label='roll_mean')
        sns.lineplot(x=date_feature, y=roll_std, data=df, label='roll_std')
        plt.title(f"Rolling statistics for '{col}' feature")
        plt.legend(loc='center right', bbox_to_anchor=(1.10, 0.5))