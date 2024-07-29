from statsmodels.tsa.stattools import adfuller
import pandas as pd


def adf_test(series):
    print('##################################')
    print('ADF Test')
    dftest = adfuller(series)
    adf = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        adf['Critical Value (%s)' % key] = value
    print(adf)
    return adf[2]