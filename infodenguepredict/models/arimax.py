"""
This is an Arima version using multiple series as predictors
"""

import numpy as np
import pandas as pd
import pyflux as pf
from datetime import datetime
import matplotlib.pyplot as plt
from infodenguepredict.data.infodengue import get_alerta_table, build_multicity_dataset





def build_model(data, ar=4, ma=4, integ=0, target=None):
    model = pf.ARIMAX(data=data,
                      # formula='{}~casos_est_min + casos_est_max+ casos+ p_rt1 + p_inc100k +nivel'.format(target),
                      ar=4,
                      formula='{}~'.format(target) + '+'.join([col for col in data.columns if col != target][:10]),
                      ma=4, integ=0)
    return model


if __name__ == "__main__":
    prediction_window = 5  # weeks
    # data = get_alerta_table(3303609)  # Nova Iguaçu: 3303609
    data = build_multicity_dataset('RJ')
    target = 'casos_est_3304557'
    data[target].plot()
    model = build_model(data, 4, 4, 1, target=target)
    fit = model.fit()  # 'BBVI',iterations=1000,optimizer='RMSProp')
    print(fit.summary())
    model.plot_fit()
    plt.savefig('arimax_in_sample.jpg')
    # model.plot_predict(h=22, oos_data=data.iloc[-22:], past_values=52)
    model.plot_predict_is(h=52, past_values=52, intervals=True)
    plt.savefig('arimax_prediction.jpg')
