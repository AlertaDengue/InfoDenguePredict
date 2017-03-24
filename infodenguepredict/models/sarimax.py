"""
This is an Arima version using multiple series as predictors
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
from datetime import datetime
import matplotlib.pyplot as plt
from infodenguepredict.data.infodengue import get_alerta_table


def build_model(data):
    model = sm.tsa.statespace.SARIMAX(endog=data.casos, exog=data[['casos_est', 'casos_est_max', 'p_inc100k', 'nivel']],
                                      order=(1, 1, 1),
                                      seasonal_order=(1, 1 , 1 , 52),
                                      time_varying_regression=True,
                                      mle_regression=False,
                                      enforce_stationarity=False
                                      )

    return model


if __name__ == "__main__":
    prediction_window = 5  # weeks
    data = get_alerta_table(3304557)  # Nova Iguaçu: 3303609
    data.casos_est.plot()
    # Graph data autocorrelation
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))

    fig = sm.graphics.tsa.plot_acf(data.ix[1:, 'casos'], lags=52, ax=axes[0])
    fig = sm.graphics.tsa.plot_pacf(data.ix[1:, 'casos'], lags=52, ax=axes[1])

    model = build_model(data)
    fit = model.fit(disp=True)  # 'BBVI',iterations=1000,optimizer='RMSProp')
    print(fit.summary())

    plt.figure()
    predict = fit.get_prediction(start='2017-01-01', dynamic=False)
    predict_ci = predict.conf_int()
    predictdy = fit.get_prediction(start='2017-01-01', dynamic=True)
    predictdy_ci = predictdy.conf_int()
    data.casos.plot(style='o',label='obs')
    predict.predicted_mean.plot(style='r--', label='one step ahead')
    predictdy.predicted_mean.plot(style='g', label='Dynamic forecast')
    plt.fill_between(predict_ci.index, predict_ci.ix[:, 0], predict_ci.ix[:, 1], color='r', alpha=0.1)
    plt.fill_between(predictdy_ci.index, predictdy_ci.ix[:, 0], predictdy_ci.ix[:, 1], color='g', alpha=0.1)
    #forecast = fit.forecast(10)
    #forecast.plot(style='b;', label='forecast')
    plt.legend(loc=0)
    plt.show()

    plt.savefig('sarimax_prediction.jpg')
