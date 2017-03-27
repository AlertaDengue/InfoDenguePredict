import numpy as np
import pandas as pd
import pyflux as pf
from datetime import datetime
import matplotlib.pyplot as plt
from infodenguepredict.data.infodengue import get_alerta_table


def build_model(data, ar=4, sc=4, family=pf.families.Poisson, formula=None):
    if formula is None:
        formula = "casos~1"
    model = pf.GASX(data=data, ar=ar, sc=sc, family=family(), formula=formula)
    return model


if __name__ == "__main__":
    prediction_window = 5  # weeks
    data = get_alerta_table(3304557)  # Nova Iguaçu: 3303609
    # print(data.info())
    # data.casos.plot()
    model = build_model(data, ar=2, sc=6)
    fit = model.fit()#'BBVI',iterations=1000,optimizer='RMSProp')
    print(fit.summary())
    model.plot_fit()
    plt.savefig('GASX_in_sample.png')
    model.plot_parameters()
    #model.plot_predict(h=10, past_values=52)
    #plt.savefig('GASX_prediction.png')
    # plt.show()
