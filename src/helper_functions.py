import datetime as dt
import logging
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import statsmodels.api as sm
import seaborn as sns
import scipy
import plotly.graph_objects as go
import warnings
import matplotlib.dates as mdates

from scipy.stats import mannwhitneyu
from scipy.stats import stats
from pandas._libs.lib import dicts_to_array
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.api import qqplot
from statsmodels.tsa.stattools import adfuller
from scipy.stats import shapiro
from scipy.stats import stats

from scrapy import Selector
from bs4 import BeautifulSoup as Soup
from pandas._libs.lib import dicts_to_array
from binance.client import Client
from binance.cm_futures import CMFutures
from binance.lib.utils import config_logging
from binance.spot import Spot as Client
from statsmodels.formula.api import ols


def transform_log(df):
    """does log transformation and
    compares original vs transformed shapiro p-values"""

    dummy_df = []

    for d in df:
        dummy_df = np.log(df)

    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].hist(df, edgecolor='black')
    axs[1].hist(dummy_df, edgecolor='black')
    axs[0].set_title('Original Data')
    axs[1].set_title('Log Transformed Data')
    plt.show()

    pvalue = shapiro(dummy_df)
    print(pvalue[1])


def transform_sqrt(df):
    """does sqrt transformation and
    compares original vs transformed shapiro p-values"""
    dummy_df = []

    for d in df:
        dummy_df = np.sqrt(df)

    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].hist(df, edgecolor='black')
    axs[1].hist(dummy_df, edgecolor='black')
    axs[0].set_title('Original Data')
    axs[1].set_title('Square Root Transformed Data')
    plt.show()
    pvalue = shapiro(dummy_df)
    print(pvalue[1])


def transform_cubic(df):
    """does cubic transformation and
    compares original vs transformed shapiro p-values"""

    dummy_df = []

    for d in df:
        dummy_df = np.cbrt(df)

    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].hist(df, edgecolor='black')
    axs[1].hist(dummy_df, edgecolor='black')
    axs[0].set_title('Original Data')
    axs[1].set_title('Cubic Transformed Data')
    plt.show()

    pvalue = shapiro(dummy_df)
    print(pvalue[1])


def transform_boxcox(df):
    """does boxcox transformation and
    compares original vs transformed shapiro p-values"""

    for d in df:
        fitted_data, lambda_data = stats.boxcox(df)

    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].hist(lambda_data, edgecolor='black')
    axs[1].hist(fitted_data, edgecolor='black')
    axs[0].set_title('Original Data')
    axs[1].set_title('Box Cox Transformed Data')
    plt.show()

    pvalue = shapiro(fitted_data)
    print(pvalue[1])


def var_normality(df1, df2):
    levene_result = scipy.stats.levene(df1, df2, center="mean")
    print("\nP value of the Levene's test is " + str(levene_result[1]))
    if levene_result[1] < 0.05:
        print("\nVariances are not equal")
    else:
        print("\nVariances are equal")


def two_ttest(df1, df2):
    t, pvalue = stats.ttest_ind(df1, df2)

    if pvalue < 0.05:
        print("\nHo is rejected")
        print("\npvalue : " + str(round(pvalue, 3)))
    else:
        print("\npvalue : " + str(round(pvalue, 3)))
        print("\nHo should be retained")


def is_normal(df):
    result = shapiro(df)
    if result[1] > 0.05:
        print("Normally Distributed")
    else:
        print("Not Normally Distributed")


def model_ols(y, x):
    """x = independent variable(s),
      y=dependent variable"""

    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    prediction = model.predict(x)
    return model.summary()


def qq(ols_df):
    sm.qqplot(data=ols_df.resid, fit=True, line="45")
    plt.show()


def adf_test(timeseries):
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def kpss_test(timeseries):
    print('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', '#Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    print(kpss_output)


def plot_arma(df):
    plot_acf(df)
    plot_pacf(df)
    plt.show()


def opt_arimax(df1, df2):
    """ df2 should be used only when there is
    exogeneous variables within the data"""
    results = pm.auto_arima(df1,
                            df2,
                            d=2,
                            start_p=1,
                            start_q=1,
                            max_p=10,
                            max_q=10,
                            information_criterion="aic",
                            trace=True,
                            error_action="ignore")
    return results


def automate_arimax(df1, df2):
    order_aic_bic = []
    for p in range(3):
        for i in range(1, 3):
            for q in range(3):
                model = ARIMA(df1, order=(p, i, q), exog=df2)
                results = model.fit()

                order_aic_bic.append(((p, i, q,), results.aic, results.bic))

    order_aic_bic = pd.DataFrame(order_aic_bic, columns=[("p", "i", "q"), "aic", "bic"])
    print(order_aic_bic.sort_values(by="aic", ascending=False).head(15))


def model_arimax(df1, df2, start, steps, exog_step):
    """models arima accordingly the given p,i,q and
    plots diagnostics of the model residuals.Start,steps and
    exog_step values should be negative (eg: if given -30, it starts
    predicting from the last 30 obsv)"""

    # builds the arima model
    model = ARIMA(df1, order=(1, 3, 1), exog=df2)
    results = model.fit()
    print(results.summary())

    # plots model diagnostics
    results.plot_diagnostics(figsize=(16, 8))

    # fitting process begins
    forecast = results.get_prediction(start=start)
    mean_forecast = forecast.predicted_mean

    # starts forecasting
    """ the steps value and the exog values should match 
    in terms of where the prediction begins at"""

    forecast2 = results.get_forecast(steps=steps, exog=df2.iloc[exog_step:])
    mean_forecast2 = forecast2.predicted_mean
    confidence_intervals2 = forecast2.conf_int()

    # plots the forecasted timeseries
    fig, ax = plt.subplots()

    plt.plot(mean_forecast2.index,
             mean_forecast2.values,
             color="black",
             label="forecasted")

    plt.show()
