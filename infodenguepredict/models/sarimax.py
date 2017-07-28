"""
This is an Arima version using multiple series as predictors
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import pickle
import statsmodels.api as sm
from datetime import datetime
import matplotlib.pyplot as plt
from infodenguepredict.data.infodengue import get_alerta_table, get_cluster_data


def build_model(data, endog, exog, **kwargs):
    model = sm.tsa.statespace.SARIMAX(endog=data[endog],
                                      exog=None if exog == [] else data[exog], #data[['casos_est', 'casos_est_max', 'p_inc100k', 'nivel']],
                                      order=(2, 1, 1),
                                      seasonal_order=(2, 1, 1, 8),
                                      time_varying_regression=True,
                                      mle_regression=False,
                                      enforce_stationarity=False,
                                      **kwargs)


    return model


if __name__ == "__main__":
    prediction_window = 5  # weeks
    city = 3304557
    state = 'RJ'

    # data = get_alerta_table(3304557)  # Nova Iguaçu: 3303609
    with open('clusters_{}.pkl'.format(state), 'rb') as fp:
        clusters = pickle.load(fp)

    data = get_cluster_data(city, clusters)
    label= 'casos_{}'.format(city)
    features = list(data.columns)
    features.remove(label)

    data[label].plot()

    # Graph data autocorrelation
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))

    fig = sm.graphics.tsa.plot_acf(data.ix[1:, label], lags=52, ax=axes[0])
    fig = sm.graphics.tsa.plot_pacf(data.ix[1:, label], lags=52, ax=axes[1])


    # model = build_model(data, 'casos', ['p_rt1'])
    model = build_model(data, label, features)
    fit = model.fit(disp=False)  # 'BBVI',iterations=1000,optimizer='RMSProp')
    print(fit.summary())

    plt.figure()
    predict = fit.get_prediction(start='2017-01-01', dynamic=False)
    predict_ci = predict.conf_int()
    # predictdy = fit.get_prediction(start='2017-01-01', dynamic=True)
    # predictdy_ci = predictdy.conf_int()
    data[label].plot(style='o',label='obs')
    predict.predicted_mean.plot(style='r--', label='one step ahead')
    # predictdy.predicted_mean.plot(style='g', label='Dynamic forecast')
    plt.fill_between(predict_ci.index, predict_ci.ix[:, 0], predict_ci.ix[:, 1], color='r', alpha=0.1)
    # plt.fill_between(predictdy_ci.index, predictdy_ci.ix[:, 0], predictdy_ci.ix[:, 1], color='g', alpha=0.1)
    plt.legend(loc=0)
    plt.savefig('sarimax_prediction.jpg')

    ## Forecast
    forecast = fit.get_prediction(start='2017-03-05', end='2017-06-21', dynamic=False)
    forecast_ci = forecast.conf_int()
    forecast.predicted_mean.plot(style='b--', label='one step ahead')
    plt.fill_between(forecast_ci.index, forecast_ci.ix[:, 0], forecast_ci.ix[:, 1], color='b', alpha=0.1)
    plt.legend(loc=0)
    plt.title('Out-of-Sample forecast')

    plt.show()


