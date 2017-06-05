"""
This is a Markov switching model version using multiple exogenous variables
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels as sm
from statsmodels.api import graphics
from datetime import datetime
import matplotlib.pyplot as plt
from infodenguepredict.data.infodengue import get_alerta_table


def build_model(data):
    model = sm.tsa.regime_switching.markov_autoregression.MarkovAutoregression(endog=data.casos.diff(), k_regimes=2,
                                                                               exog=data[['p_rt1', 'p_inc100k']],
                                                                                order=2)


    return model


if __name__ == "__main__":
    prediction_window = 5  # weeks
    data = get_alerta_table(3304557)  # Nova Iguaçu: 3303609
    data.casos_est.plot()
    # Graph data autocorrelation
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))

    fig = graphics.tsa.plot_acf(data.ix[1:, 'casos'], lags=52, ax=axes[0])
    fig = graphics.tsa.plot_pacf(data.ix[1:, 'casos'], lags=52, ax=axes[1])

    model = build_model(data)
    fit = model.fit()  # 'BBVI',iterations=1000,optimizer='RMSProp')
    print(fit.summary())
    #todo: fix model, not fitting
    plt.figure()
    predict = fit.get_prediction(start='2017-01-01', end='2017-03-01', dynamic=False)
    predict_ci = predict.conf_int()
    predictdy = fit.get_prediction(start='2017-01-01', end='2017-02-26', dynamic=True)
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
