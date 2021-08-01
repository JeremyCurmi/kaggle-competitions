import numpy as np
import pandas as pd


def log_scaling(feature: pd.Series) -> pd.Series:
    '''

    :param feature: numerical feature
    :return:
    '''
    return np.log1p(feature)


def inverse_log_scaling(feature: pd.Series) -> pd.Series:
    '''

    :param feature: numerical feature
    :return:
    '''
    return np.exp(feature) - 1