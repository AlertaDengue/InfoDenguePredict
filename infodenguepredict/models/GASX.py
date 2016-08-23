import numpy as np
import pandas as pd
import pyflux as pf
from datetime import datetime
import matplotlib.pyplot as plt
from infodenguepredict.data.infodengue import get_alerta_table


def build_model(data, ar=4, sc=4, family=pf.GASPoisson, target=None):
    model = pf.GASX(data=data, ar=ar, sc=sc, family=family(), formula="{}~casos_est_min + casos_est_max+ casos+ p_rt1 + p_inc100k +nivel".format(target))
    return model


if __name__ == "__main__":
    prediction_window = 5  # weeks
    data = get_alerta_table(3303609)  # Nova Iguaçu: 3303609
    data.casos_est.plot()
    model = build_model(data, ar=4, sc=4, target='casos_est')
    fit = model.fit()#'BBVI',iterations=1000,optimizer='RMSProp')
    print(fit.summary())
    model.plot_fit()
    plt.savefig('GAS_in_sample.svg')
    model.plot_predict(h=52, past_values=104)
    plt.savefig('GAS_prediction.svg')
    # plt.show()
