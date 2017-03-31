import numpy as np
import pandas as pd
import pyflux as pf
from datetime import datetime
import matplotlib.pyplot as plt
from infodenguepredict.data.infodengue import get_alerta_table


def build_model(data, ar=2, sc=4, family=pf.families.Poisson, target=None):
    model = pf.GASLLEV(data=data, family=family(), target=target)
    return model


if __name__ == "__main__":
    prediction_window = 5  # weeks
    data = get_alerta_table(3304557)  # Nova Iguaçu: 3303609
    data.casos.plot()
    model = build_model(data, ar=2, sc=6, target='casos')
    fit = model.fit()#'BBVI',iterations=1000,optimizer='RMSProp')
    print(fit.summary())
    model.plot_fit()
    model.plot_z(figsize=(15, 5))
    plt.savefig('GASLLEV_in_sample.png')
    data.casos.plot(style='ko')
    model.plot_predict(h=10, past_values=52)
    plt.savefig('GASLLEV_prediction.png')
    # plt.show()
