import numpy as np
import pandas as pd
import pyflux as pf
from datetime import datetime
import matplotlib.pyplot as plt
from infodenguepredict.data.infodengue import get_alerta_table


def build_model(data, ar=4, kernel=pf.SquaredExponential, target=None):
    model = pf.GPNARX(data=data, ar=ar, kernel=kernel())
    return model


if __name__ == "__main__":
    prediction_window = 5  # weeks
    data = get_alerta_table(3304557)  # Nova Iguaçu: 3303609
    data.casos.plot()
    model = build_model(data.casos.values, ar=12, target='casos')
    fit = model.fit()#'BBVI',iterations=1000,optimizer='RMSProp')
    print(fit.summary())
    model.plot_fit()
    plt.savefig('GPNARX_in_sample.svg')
    model.plot_predict(h=52, past_values=16)
    plt.savefig('GPNARX_prediction.svg')
    # plt.show()
