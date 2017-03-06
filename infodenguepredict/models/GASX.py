import numpy as np
import pandas as pd
import pyflux as pf
from datetime import datetime
import matplotlib.pyplot as plt
from infodenguepredict.data.infodengue import get_alerta_table


def build_model(data, ar=4, sc=4, family=pf.families.Poisson, target=None):
    model = pf.GASX(data=data, ar=ar, sc=sc, family=family(), formula="{}~ casos+ p_rt1 + p_inc100k +nivel".format(target))
    return model


if __name__ == "__main__":
    prediction_window = 5  # weeks
    data = get_alerta_table(3303500)  # Nova Iguaçu: 3303609
    print(data.info())
    data.casos.plot()
    model = build_model(data, ar=12, sc=6, target='casos')
    fit = model.fit()#'BBVI',iterations=1000,optimizer='RMSProp')
    print(fit.summary())
    model.plot_fit()
    plt.savefig('GASX_in_sample.png')
    model.plot_predict(h=52, past_values=104)
    plt.savefig('GASX_prediction.png')
    # plt.show()
