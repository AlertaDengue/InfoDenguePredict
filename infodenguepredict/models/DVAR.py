"""
Dynamic Vector Autogregression using statsmodels
http://statsmodels.sourceforge.net/devel/vector_ar.html
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.api import *
# from statsmodels.tsa.vector_ar.var_model import
from datetime import datetime
import matplotlib.pyplot as plt
from infodenguepredict.data.infodengue import get_alerta_table, build_multicity_dataset


def build_model(data, lag_order, window_type):
    data.index = pd.DatetimeIndex(data.index)
    model = DynamicVAR(data, lag_order=2, window=12, window_type=window_type)
    return model


if __name__ == "__main__":
    prediction_window = 2  # weeks
    scenario = 'local'
    # scenario = 'global'
    if scenario == 'local':
        data = get_alerta_table(3303500)  # Nova Iguaçu: 3303500
        data = data[['casos', 'nivel', 'p_rt1']]
    else:
        data = build_multicity_dataset('RJ')
        data = data[[col for col in data.columns if col.startswith('casos') and not col.startswith('casos_est')][:3]]
        # data = data.diff()
    print(data.info())
    #TODO: Apply Seasonal differencing to series
    # data.casos.plot(title="Series")
    model = build_model(data, 12, 'expanding')
    # fit = model.fit(maxlags=11, ic='aic') # 4 lags
    # print(model.coefs.minor_xs('casos_3303500').info())



    forecast = model.forecast(prediction_window)
    print(forecast)
    model.plot_forecast(prediction_window)
    plt.savefig('DVAR_forecast_{}_weeks.png'.format(prediction_window))
    plt.show()
