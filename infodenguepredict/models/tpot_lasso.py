import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

from infodenguepredict.data.infodengue import get_cluster_data, get_city_names
from infodenguepredict.models.random_forest import build_lagged_features
from infodenguepredict.predict_settings import *


def plot_prediction(preds, ydata, title, train_size):
    plt.clf()
    plt.plot(ydata, 'k-')

    point = ydata.index[train_size]

    min_val = min([min(ydata), np.nanmin(preds)])
    max_val = max([max(ydata), np.nanmax(preds)])
    plt.vlines(point, min_val, max_val, 'g', 'dashdot', lw=2)

    pred_window = preds.shape[1]
    llist = range(len(ydata.index) - (preds.shape[1]))
    for n in llist:
        plt.plot(ydata.index[n: n + pred_window], preds[n], 'r-.', alpha=0.3)
        plt.vlines(ydata.index[n: n + pred_window], np.zeros(pred_window), preds[n], 'b', alpha=0.2)

    plt.text(point, 0.6 * max_val, "Out of sample Predictions")
    plt.grid()
    plt.ylabel('indices')
    plt.legend(loc=0)
    plt.title('Predictions for {}'.format(title))
    plt.xticks(rotation=70)
    plt.legend(['data', 'predicted'])

    plt.savefig('{}/{}/lasso_{}.png'.format('saved_models/lasso', STATE, title), dpi=300)
    return None


def calculate_metrics(pred, ytrue):
    negs = np.where(pred < 0)[0]
    if len(negs) > 0:
        ytrue_new = ytrue.reset_index().drop(negs)
        pred_new = np.delete(pred, negs)
        msle = mean_squared_log_error(ytrue_new.drop('index', axis=1), pred_new)
    else:
        msle = mean_squared_log_error(ytrue, pred)
    return [mean_absolute_error(ytrue, pred), explained_variance_score(ytrue, pred),
            mean_squared_error(ytrue, pred), msle,
            median_absolute_error(ytrue, pred), r2_score(ytrue, pred)]


def lasso_state_prediction(state, lookback, horizon, predictors):
    clusters = pd.read_pickle('../analysis/clusters_{}.pkl'.format(state))

    for cluster in clusters:
        data_full, group = get_cluster_data(geocode=cluster[0], clusters=clusters,
                                            data_types=DATA_TYPES, cols=predictors)
        for city in cluster:
            if os.path.isfile('saved_models/lasso/{}/lasso_metrics_{}.pkl'.format(state, city)):
                print(city, 'done')
                continue

            target = 'casos_est_{}'.format(city)
            casos_est_columns = ['casos_est_{}'.format(i) for i in group]
            casos_columns = ['casos_{}'.format(i) for i in group]

            data = data_full.drop(casos_columns, axis=1)
            data_lag = build_lagged_features(data, lookback)
            data_lag.dropna()
            targets = {}
            for d in range(1, horizon + 1):
                if d == 1:
                    targets[d] = data_lag[target].shift(-(d - 1))
                else:
                    targets[d] = data_lag[target].shift(-(d - 1))[:-(d - 1)]

            X_data = data_lag.drop(casos_est_columns, axis=1)
            X_train, X_test, y_train, y_test = train_test_split(X_data, data_lag[target],
                                                                train_size=0.7, test_size=0.3, shuffle=False)

            city_name = get_city_names([city, 0])[0][1]
            preds = np.empty((len(data_lag), horizon))
            metrics = pd.DataFrame(index=('mean_absolute_error', 'explained_variance_score',
                                          'mean_squared_error', 'mean_squared_log_error',
                                          'median_absolute_error', 'r2_score'))
            for d in range(1, horizon + 1):
                model = LassoLarsCV(max_iter=15, n_jobs=-1, normalize=False)

                tgt = targets[d][:len(X_train)]
                tgtt = targets[d][len(X_train):]
                model.fit(X_train, tgt)
                pred = model.predict(X_data[:len(targets[d])])

                dif = len(data_lag) - len(pred)
                if dif > 0:
                    pred = list(pred) + ([np.nan] * dif)
                preds[:, (d - 1)] = pred
                pred_m = model.predict(X_test[:(len(tgtt))])
                metrics[d] = calculate_metrics(pred_m, tgtt)

            metrics.to_pickle('{}/{}/lasso_metrics_{}.pkl'.format('saved_models/lasso', state, city))
            plot_prediction(preds, targets[1], city_name, len(X_train))
            # plt.show()
    return None

# NOTE: Make sure that the class is labeled 'target' in the data file


if __name__=="__main__":

    for STATE in ['RJ', 'PR', 'Ceará']:
        lasso_state_prediction(STATE, LOOK_BACK, PREDICTION_WINDOW, PREDICTORS)