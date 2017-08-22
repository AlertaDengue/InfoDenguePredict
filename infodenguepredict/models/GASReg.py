import numpy as np
import pandas as pd
import pyflux as pf
from datetime import datetime
import matplotlib.pyplot as plt
from infodenguepredict.data.infodengue import combined_data


def build_model(data, family=pf.families.Laplace, formula=None):
    if formula is None:
        formula = "casos~temp_min"
    model = pf.GASReg(data=data, family=family(), formula=formula)
    return model


if __name__ == "__main__":
    city = 3304557
    prediction_window = 5  # weeks
    # data = get_alerta_table(city)  # Nova Iguaçu: 3303609
    # Fetching exogenous vars
    # T = get_temperature_data(city)  # (3303500)
    # T = T[~T.index.duplicated()]
    # Tw = get_tweet_data(city)
    # Tw = Tw[~Tw.index.duplicated()]
    Full = combined_data(city)#data.join(T.resample('W-SUN').mean()).join(Tw.resample('W-SUN').sum()).dropna()
    # print(data.info())
    # data.casos.plot()
    # print(Full.info())
    # print(Full.describe())

    # Full.to_csv('data.csv.gz', compression='gzip')
    Full.columns = [cn.replace('_','') for cn in Full.columns]
    model = build_model(Full.dropna(), formula='casos~numero+tempmin+umidmin')
    fit = model.fit()#'BBVI', iterations=1000, optimizer='RMSProp')

    print(fit.summary())
    model.plot_fit(intervals=False)
    plt.savefig('GASReg_in_sample.png')
    model.plot_parameters()
    model.plot_predict(h=5, past_values=12)
    plt.savefig('GASReg_prediction.png')
    plt.show()
