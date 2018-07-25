import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

from infodenguepredict.data.infodengue import get_cluster_data, get_city_names, combined_data, get_alerta_table
from infodenguepredict.models.random_forest import build_lagged_features
from infodenguepredict.predict_settings import *


def get_cities_from_state(state):
    alerta_table = get_alerta_table(state=state)
    cities_list = alerta_table.municipio_geocodigo.unique()
    return cities_list


def plot_prediction(preds, ydata, title, train_size, path='lasso'):
    plt.clf()
    plt.plot(ydata, 'k-')

    point = ydata.index[train_size]

    min_val = min([min(ydata), np.nanmin(preds)])
    max_val = max([max(ydata), np.nanmax(preds)])
    plt.vlines(point, min_val, max_val, 'g', 'dashdot', lw=2)

    pred_window = preds.shape[1]
    llist = range(len(ydata.index) - (preds.shape[1]))

    # for figure with only the last prediction point (single red line)
    x = []
    y = []
    for n in llist:
        plt.vlines(ydata.index[n + pred_window], 0, preds[n][-1], 'b', alpha=0.2)
        x.append(ydata.index[n + pred_window])
        y.append(preds[n][-1])
    plt.plot(x, y, 'r-', alpha=0.7)

    # # for figure with all predicted points
    # for n in llist:
    #     plt.vlines(ydata.index[n + pred_window], 0, preds[n][-1], 'b', alpha=0.2)
    #     plt.plot(ydata.index[n: n + pred_window], preds[n], 'r-.', alpha=0.3)
    #     # plt.vlines(ydata.index[n: n + pred_window], np.zeros(pred_window), preds[n], 'b', alpha=0.2)

    plt.text(point, 0.6 * max_val, "Out of sample Predictions")
    plt.grid()
    plt.ylabel('indices')
    plt.legend(loc=0)
    plt.title('Predictions for {}'.format(title))
    plt.xticks(rotation=70)
    plt.legend(['data', 'predicted'])

    plt.savefig('saved_models/{}/{}/lasso_{}_ss.png'.format(path, STATE, title), dpi=300)
    return None


# def mape(ytrue, pred):
#     if 0 in pred:
#
#     return np.mean(np.abs((ytrue - pred)/ytrue)) * 100


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


def lasso_single_prediction(city, state, lookback, horizon, predictors):
    clusters = pd.read_pickle('../analysis/clusters_{}.pkl'.format(state))
    data, group = get_cluster_data(geocode=city, clusters=clusters,
                                        data_types=DATA_TYPES, cols=predictors)

    target = 'casos_est_{}'.format(city)
    casos_est_columns = ['casos_est_{}'.format(i) for i in group]
    # casos_columns = ['casos_{}'.format(i) for i in group]

    # data = data_full.drop(casos_columns, axis=1)
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

    if sum(y_train)==0:
        print('aaaah',city)
        return None
    city_name = get_city_names([city, 0])[0][1]
    preds = np.empty((len(data_lag), horizon))
    metrics = pd.DataFrame(index=('mean_absolute_error', 'explained_variance_score',
                                  'mean_squared_error', 'mean_squared_log_error',
                                  'median_absolute_error', 'r2_score'))
    for d in range(1, horizon + 1):
        model = LassoLarsCV(max_iter=5, n_jobs=-1, normalize=False)

        tgt = targets[d][:len(X_train)]
        tgtt = targets[d][len(X_train):]
        try:
            model.fit(X_train, tgt)
            print(city, 'done')
        except ValueError as err:
            print('-----------------------------------------------------')
            print(city, 'ERRO')
            print('-----------------------------------------------------')
            break
        pred = model.predict(X_data[:len(targets[d])])

        dif = len(data_lag) - len(pred)
        if dif > 0:
            pred = list(pred) + ([np.nan] * dif)
        preds[:, (d - 1)] = pred
        pred_m = model.predict(X_test[:(len(tgtt))])
        metrics[d] = calculate_metrics(pred_m, tgtt)

    metrics.to_pickle('{}/{}/lasso_metrics_{}.pkl'.format('saved_models/lasso', state, city))
    plot_prediction(preds, targets[1], city_name, len(X_train))
    return None


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
                try:
                    model.fit(X_train, tgt)
                except ValueError as err:
                    print('-----------------------------------------------------')
                    print(city, 'ERRO')
                    print('-----------------------------------------------------')
                    break
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


def lasso_single_state_prediction(state, lookback, horizon, predictors):
    ##LASSO WITHOUT CLUSTER SERIES
    cities = list(get_cities_from_state('Ceará'))

    for city in cities:
        if os.path.isfile('/home/elisa/Documentos/InfoDenguePredict/infodenguepredict/models/saved_models/lasso_no_cluster/{}/lasso_metrics_{}.pkl'.format(state, city)):
            print(city, 'done')
            continue
        data = combined_data(city, DATA_TYPES)
        data = data[predictors]
        data.drop('casos', axis=1, inplace=True)

        target = 'casos_est'
        data_lag = build_lagged_features(data, lookback)
        data_lag.dropna()
        targets = {}
        for d in range(1, horizon + 1):
            if d == 1:
                targets[d] = data_lag[target].shift(-(d - 1))
            else:
                targets[d] = data_lag[target].shift(-(d - 1))[:-(d - 1)]

        X_data = data_lag.drop(target, axis=1)
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
            try:
                model.fit(X_train, tgt)
            except ValueError as err:
                print('-----------------------------------------------------')
                print(city, 'ERRO')
                print('-----------------------------------------------------')
                break
            pred = model.predict(X_data[:len(targets[d])])

            dif = len(data_lag) - len(pred)
            if dif > 0:
                pred = list(pred) + ([np.nan] * dif)
            preds[:, (d - 1)] = pred
            pred_m = model.predict(X_test[:(len(tgtt))])
            metrics[d] = calculate_metrics(pred_m, tgtt)

            metrics.to_pickle('{}/{}/lasso_metrics_{}.pkl'.format('saved_models/lasso_no_cluster', state, city))
        plot_prediction(preds, targets[1], city_name, len(X_train), path='lasso_no_cluster')
        # plt.show()
    return None

# NOTE: Make sure that the class is labeled 'target' in the data file


if __name__ == "__main__":

    lasso_single_prediction(CITY, STATE, LOOK_BACK, PREDICTION_WINDOW, predictors=PREDICTORS)

    # for STATE in ['CE']:
    #     lasso_single_state_prediction(STATE, LOOK_BACK, PREDICTION_WINDOW, PREDICTORS)
    # lasso_single_prediction(4111704, 'PR', LOOK_BACK, PREDICTION_WINDOW, PREDICTORS)
