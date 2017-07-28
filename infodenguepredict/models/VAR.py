"""
Vector Autogregression based on examples from
http://www.pyflux.com/bayesian-vector-autoregression/
"""

import numpy as np
import pandas as pd
import pyflux as pf
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from infodenguepredict.data.infodengue import get_alerta_table, get_cluster_data


def build_model(data, lags):
    model = pf.VAR(data=data, lags=lags)
    return model


if __name__ == "__main__":
    prediction_window = 5  # weeks
    city = 3304557
    state = 'RJ'

    with open('clusters_{}.pkl'.format(state), 'rb') as fp:
        clusters = pickle.load(fp)

    # data = build_multicity_dataset('RJ')
    # data = get_alerta_table(3304557) # Nova Iguaçu: 3303500
    data = get_cluster_data(city, clusters)

    for col in list(filter(lambda x: 'casos' in x, data.columns)):
        data[col] = data[col].astype('float')
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
