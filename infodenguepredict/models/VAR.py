"""
Vector Autogregression based on examples from
http://www.pyflux.com/bayesian-vector-autoregression/
"""

import numpy as np
import pandas as pd
import pyflux as pf
from datetime import datetime
import matplotlib.pyplot as plt
from infodenguepredict.data.infodengue import get_alerta_table


def build_model(data, lags):
    model = pf.VAR(data=data, lags=lags)
    return model


if __name__ == "__main__":
    prediction_window = 5  # weeks
    data = get_alerta_table(3303609)  # Nova Iguaçu: 3303609
    data = data[['casos_est', 'p_rt1', 'p_inc100k']]
    # print(data.info())
    data.casos_est.plot()
    model = build_model(data, lags=4)
    fit = model.fit()#'BBVI',iterations=1000,optimizer='RMSProp')
    print(fit.summary())
    model.plot_fit()
    plt.savefig('VAR_in_sample.svg')
    model.plot_predict(h=52, past_values=104)
    plt.savefig('VAR_prediction.svg')
    # plt.show()
