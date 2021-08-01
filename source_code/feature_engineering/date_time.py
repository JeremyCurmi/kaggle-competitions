import pandas as pd


def get_season_from_date_time_feature(date_time: pd.Series) -> pd.Series:
    '''

    :param date_time: datetime feature
    :return:
    '''
    month_feature = date_time.dt.month
    return month_feature%12 // 3 + 1