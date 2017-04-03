import numpy as np
import pandas as pd
import pyflux as pf
from datetime import datetime
import matplotlib.pyplot as plt
from infodenguepredict.data.infodengue import get_alerta_table, get_temperature_data, get_tweet_data, get_rain_data


def build_model(data, ar=4, sc=4, family=pf.families.Poisson, formula=None):
    if formula is None:
        formula = "casos~1"
    model = pf.GASX(data=data, ar=ar, sc=sc, family=family(), formula=formula)
    return model


if __name__ == "__main__":
    prediction_window = 5  # weeks
    data = get_alerta_table(3304557)  # Nova Iguaçu: 3303609
    # Fetching exogenous vars
    T = get_temperature_data(3304557)  # (3303500)
    T = T[~T.index.duplicated()]
    Tw = get_tweet_data(3304557)
    Tw = Tw[~Tw.index.duplicated()]
    Full = data.join(T.resample('W-SUN').mean()).join(Tw.resample('W-SUN').sum()).dropna()
    # print(data.info())
    # data.casos.plot()
    #print(Full.info())
    model = build_model(Full, ar=4, sc=6, formula='casos~1+temp_min+casos_est+p_inc100k+numero+umid_min+pressao_min+numero')
    fit = model.fit('Laplace')# 'BBVI', iterations=1000, optimizer='RMSProp')

    print(fit.summary())
    model.plot_fit()
    plt.savefig('GASX_in_sample.png')
    model.plot_parameters()
    #model.plot_predict(h=10, past_values=52)
    #plt.savefig('GASX_prediction.png')
    # plt.show()
