import numpy as np
import pandas as pd
import pyflux as pf
from datetime import datetime
import matplotlib.pyplot as plt
from infodenguepredict.data.infodengue import get_alerta_table


def build_model(data, ar=4, sc=4, family=pf.families.Poisson, target=None):
    model = pf.GAS(data=data, ar=ar, sc=sc, family=family(), target=target)
    return model


if __name__ == "__main__":
    prediction_window = 5  # weeks
    data = get_alerta_table(3303609)  # Nova Iguaçu: 3303609
    data.casos.plot()
    model = build_model(data, ar=12, sc=6, target='casos')
    fit = model.fit()#'BBVI',iterations=1000,optimizer='RMSProp')
    print(fit.summary())
    model.plot_fit()
    plt.savefig('GAS_in_sample.png')
    model.plot_predict(h=52, past_values=104)
    plt.savefig('GAS_prediction.png')
    # plt.show()
