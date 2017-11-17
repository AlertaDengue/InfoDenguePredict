import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from datetime import datetime
import matplotlib.pyplot as plt
from infodenguepredict.data.infodengue import get_alerta_table


def build_model(data, **kwargs):
    model = RandomForestRegressor(max_depth=None, random_state=0)
    return model

def rolling_forecasts(model, window=12, horizon=1):
    pass
    # TODO: implement the training of the model using a rolling window

if __name__ == "__main__":
    prediction_window = 5  # weeks
    data = get_alerta_table(3304557)  # Nova Iguaçu: 3303609
    data.casos_est.plot()
    model = build_model(data)
    data2 = data.pop('casos_est')
    model.fit(data2.values, data.casos.values)

    print(model.feature_importances_)

