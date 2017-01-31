"""
Vector Autogregression using statsmodels
http://statsmodels.sourceforge.net/devel/vector_ar.html
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.api import *
# from statsmodels.tsa.vector_ar.var_model import
from datetime import datetime
import matplotlib.pyplot as plt
from infodenguepredict.data.infodengue import get_alerta_table, build_multicity_dataset


def build_model(data):
    data.index = pd.DatetimeIndex(data.index)
    model = VAR(data)
    return model


if __name__ == "__main__":
    prediction_window = 5  # weeks
    data = get_alerta_table(3303609)  # Nova Iguaçu: 3303609
    data = data[['casos_est', 'nivel']]
    # data = build_multicity_dataset('RJ')
    #data = data[[col for col in data.columns if col.startswith('nivel')]]
    # print(data.info())
    # data.casos_est.plot(title="Series")
    model = build_model(data)
    fit = model.fit(4) # 4 lags
    print(fit.summary())
    fit.plot()
    fit.plot_acorr()

    plt.figure()
    lag_order = fit.k_ar
    forecast = fit.forecast(data.values[-lag_order:], 5)
    print (forecast)
    # var.plot_forecast(forecast)

    plt.show()
