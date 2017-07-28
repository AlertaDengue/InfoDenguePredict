"""
Vector Autogregression based on examples from
http://www.pyflux.com/bayesian-vector-autoregression/
"""

import numpy as np
import pandas as pd
import pyflux as pf
from datetime import datetime
import matplotlib.pyplot as plt
from infodenguepredict.data.infodengue import get_alerta_table, build_multicity_dataset


def build_model(data, lags):
    model = pf.VAR(data=data, lags=lags)
    return model


if __name__ == "__main__":
    prediction_window = 5  # weeks
    # data = get_alerta_table(3304557) # Nova Iguaçu: 3303500
    data = build_multicity_dataset('RJ')
    data = data[[c for c in data.columns if c.startswith("casos_") and not c.startswith('casos_est')][:10]]
    data = data.astype(float)
    # data.casos = data.casos.astype('float')
    # data = data[['casos', 'nivel']]
    # print(data.info())
    # data.casos.plot(title="series")
    model = build_model(data, lags=12)
    fit = model.fit()#'BBVI',iterations=1000, optimizer='RMSProp')
    print(fit.summary())
    model.plot_fit()
    plt.savefig('VAR_in_sample.png')
    model.plot_predict(h=5, past_values=104)
    plt.savefig('VAR_prediction.png')
    print(model.predict(h=15))
    model.plot_predict_is(h=15)
    # plt.show()
