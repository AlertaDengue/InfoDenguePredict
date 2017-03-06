"""
Model using Facebook's prophet Library
reference: https://facebookincubator.github.io/prophet/static/prophet_paper_20170113.pdf
"""

import pandas as pd
import numpy as np
from fbprophet import Prophet
from infodenguepredict.data.infodengue import get_alerta_table
from matplotlib import pyplot as P

def build_model():
    return Prophet()

if __name__ == "__main__":
    prediction_window = 90  # days
    data = get_alerta_table(3304557)  # Nova Iguaçu: 3303609
    # print(data.info())
    Model = build_model()
    df = pd.DataFrame()
    df['ds'] = data.index.values
    df['y'] = data.casos.values
    #df = df.dropna()
    # print(data.casos)
    # print(df.info())
    Model.fit(df)

    future = Model.make_future_dataframe(periods=prediction_window)
    print(future)
    forecast = Model.predict(future)
    Model.plot(forecast);
    P.show()

