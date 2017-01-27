import numpy as np
import pandas as pd
import pyflux as pf
from datetime import datetime
import matplotlib.pyplot as plt
from infodenguepredict.data.infodengue import get_alerta_table


def build_model(data, ar=4, ma=4, integ=0, target=None):
    model = pf.ARIMA(data=data, ar=4, ma=4, integ=0, target=target)
    return model


if __name__ == "__main__":
    prediction_window = 5  # weeks
    data = get_alerta_table(3303609)  # Nova Iguaçu: 3303609
    data.casos_est.plot()
    model = build_model(data, 4, 4, 1, 'casos')
    print(model.latent_variables)
    model.adjust_prior(0, pf.Normal(2,1))
    fit = model.fit('BBVI', iterations=1000, optimizer='RMSProp')
    print(fit.summary())
    model.plot_fit()
    plt.savefig('arima_in_sample.svg')
    model.plot_predict(h=52, past_values=52)
    plt.savefig('arima_prediction.svg')
