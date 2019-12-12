"""
This scripts implements cross disease predicitons using RQF model trained on dengue
"""

from infodenguepredict.models.quantile_forest import build_model, build_lagged_features, calculate_metrics
from infodenguepredict.data.infodengue import get_cluster_data, get_city_names
from infodenguepredict.predict_settings import *
import joblib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from skgarden import RandomForestQuantileRegressor


def plot_prediction(pred, pred25, pred975, ydata, horizon, title, path='quantile_forest', save=True, doenca='chik'):
    plt.clf()
    plt.plot(ydata, 'k-', label='data')

    x = ydata.index.shift(horizon, freq='W')
    plt.plot(x, pred, 'r-', alpha=0.5, label='median prediction')
    # plt.plot(x, y25, 'b-', alpha=0.3)
    # plt.plot(x, y975, 'b-', alpha=0.3)
    plt.fill_between(x, pred25, pred975, color='b', alpha=0.3)

    plt.grid()
    plt.ylabel('Weekly cases')
    plt.title('{} cross-predictions for {}'.format(doenca, title))
    plt.xticks(rotation=70)
    plt.legend(loc=0)
    if save:
        if not os.path.exists('saved_models/' + path + '/' + STATE):
            os.mkdir('saved_models/' + path + '/' + STATE)

        plt.savefig('saved_models/{}/{}/qf_{}_cross_{}_.png'.format(path, STATE, doenca, title), dpi=300)
    plt.show()
    return None


def qf_prediction(city, state, horizon, lookback, doenca='chik'):
    with open('../analysis/clusters_{}.pkl'.format(state), 'rb') as fp:
        clusters = pickle.load(fp)
    data, group = get_cluster_data(city, clusters=clusters, data_types=DATA_TYPES, cols=PREDICTORS, doenca=doenca)

    target = 'casos_est_{}'.format(city)
    casos_est_columns = ['casos_est_{}'.format(i) for i in group]
    # casos_columns = ['casos_{}'.format(i) for i in group]

    # data = data_full.drop(casos_columns, axis=1)
    data_lag = build_lagged_features(data, lookback)
    data_lag.dropna()
    data_lag = data_lag['2016-01-01':]
    targets = {}
    for d in range(1, horizon + 1):
        if d == 1:
            targets[d] = data_lag[target].shift(-(d - 1))
        else:
            targets[d] = data_lag[target].shift(-(d - 1))[:-(d - 1)]

    X_data = data_lag.drop(casos_est_columns, axis=1)

    city_name = get_city_names([city, 0])[0][1]

    #  Load dengue model
    model = joblib.load('saved_models/quantile_forest/{}/{}_city_model_{}W.joblib'.format(state, city, horizon))
    pred25 = model.predict(X_data, quantile=2.5)
    pred = model.predict(X_data, quantile=50)
    pred975 = model.predict(X_data, quantile=97.5)

    # metrics.to_pickle('{}/{}/qf_metrics_{}.pkl'.format('saved_models/quantile_forest', state, city))
    plot_prediction(pred, pred25, pred975, targets[1], horizon, city_name, save=True, doenca=doenca)

    return model, pred, pred25, pred975, X_data, targets, data_lag


if __name__ == "__main__":
    doença = 'chik'
    STATE = 'RJ'
    if STATE == 'RJ':
        cities = [3304557, 3303500, 3301009, 3304904]
    elif STATE == 'CE':
        cities = [2304400, 2307650]
    for CITY in cities:
        model, preds, preds25, preds975, X_data, targets, data_lag = qf_prediction(CITY, STATE,
                                                                                   horizon=PREDICTION_WINDOW,
                                                                                   lookback=LOOK_BACK, doenca=doença)
        # Save cross-predictions
        with open(f'saved_models/quantile_forest/{STATE}/{CITY}_cross_{doença}_preditions.pkl','wb') as f:
            pickle.dump({'xdata': X_data, 'target': targets, 'pred': preds, 'ub': preds975, 'lb': preds25}, f)
