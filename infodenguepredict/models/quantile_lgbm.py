import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import pickle
from joblib import dump, load
from collections import defaultdict

import matplotlib.pyplot as plt
from infodenguepredict.data.infodengue import get_cluster_data, get_city_names, combined_data, get_alerta_table
from infodenguepredict.predict_settings import *

import lightgbm as lgb
import optuna.integration.lightgbm as lgbt  # Tuner for lgbm


def get_cities_from_state(state):
    alerta_table = get_alerta_table(state=state)
    cities_list = alerta_table.municipio_geocodigo.unique()
    return cities_list


def build_model(alpha=0.5, params=None, **kwargs):
    '''
    Return an LGBM model for the quantile specified in alpha
    :param alpha: quantile to regress for,
    :param kwargs:
    :return: LGBMRegressor model
    '''
    if params is None:
        params = {
            'n_jobs': 4,
            'max_depth': 4,
            'max_bin': 63,
            'num_leaves': 255,
            'min_data_in_leaf': 1,
            'subsample': 0.9,
            'n_estimators': 80,
            'learning_rate': 0.1,
            'colsample_bytree': 0.9,
            'boosting_type': 'gbdt'
        }

    model = lgb.LGBMRegressor(objective='quantile', alpha=alpha, **params)

    return model


def build_lagged_features(dt, lag=2, dropna=True):
    '''
    Builds a new DataFrame to facilitate regressing over all possible lagged features
    :param dt: Dataframe containing features
    :param lag: maximum lags to compute
    :param dropna: if true the initial rows containing NANs due to lagging will be dropped
    :return: Dataframe
    '''
    if type(dt) is pd.DataFrame:
        new_dict = {}
        for col_name in dt:
            new_dict[col_name] = dt[col_name]
            # create lagged Series
            for l in range(1, lag + 1):
                new_dict['%s_lag%d' % (col_name, l)] = dt[col_name].shift(l)
        res = pd.DataFrame(new_dict, index=dt.index)

    elif type(dt) is pd.Series:
        the_range = range(lag + 1)
        res = pd.concat([dt.shift(-i) for i in the_range], axis=1)
        res.columns = ['lag_%d' % i for i in the_range]
    else:
        print('Only works for DataFrame or Series')
        return None
    if dropna:
        return res.dropna()
    else:
        return res


def build_and_fit(data, target, quantile=0.5, tune=False):
    """
    Builds and Fits the model
    :param data: feature Dataframe
    :param target: variable to be forecasted
    :param tune: if True, tune hyperparameters first.
    :return:
    """
    if tune:
        tuned_params = tune_hyperparam(data, target, quantile)
    else:
        tuned_params = None
    model = build_model(alpha=quantile, params=tuned_params)
    model.fit(data.values, target)

    return model


def tune_hyperparam(data, target, quantile=0.5):
    params = {
        'n_jobs': 4,
        'max_depth': 4,
        'objective': 'quantile',
        'subsample': 0.9,
        'n_estimators': 80,
        'learning_rate': 0.1,
        'colsample_bytree': 0.9,
        'boosting_type': 'gbdt'
    }
    train_x, val_x, train_y, val_y = train_test_split(data, target, test_size=0.25)
    dtrain = lgb.Dataset(train_x, label=train_y)
    dval = lgb.Dataset(val_x, label=val_y)

    booster = lgbt.train(params, dtrain,
                         valid_sets=[dtrain, dval],
                         verbose_eval=100,
                         )
    best_params = booster.params
    return best_params


def calculate_metrics(pred, ytrue):
    return [mean_absolute_error(ytrue, pred, ), explained_variance_score(ytrue, pred),
            mean_squared_error(ytrue, pred),
            median_absolute_error(ytrue, pred), r2_score(ytrue, pred)]


def plot_prediction(preds, preds25, preds975, ydata, title, train_size, path='quantile_lgbm', save=True):
    plt.clf()
    plt.plot(ydata, 'k-', label='data')

    point = ydata.index[train_size]

    min_val = min([min(ydata), np.nanmin(preds)])
    max_val = max([max(ydata), np.nanmax(preds)])
    plt.vlines(point, min_val, max_val, 'g', 'dashdot', lw=2)

    pred_window = preds.shape[1]
    llist = range(len(ydata.index) - (preds.shape[1]))
    # print(type(preds))

    # # for figure with all predicted points
    # for n in llist:
    #     plt.vlines(ydata.index[n + pred_window], 0, preds[n][-1], 'b', alpha=0.2)
    #     plt.plot(ydata.index[n:n + pred_window], preds[n], 'r')

    # for figure with only the last prediction point (single red line)
    x = []
    y = []
    y25 = []
    y975 = []
    for n in llist:
        # plt.vlines(ydata.index[n + pred_window], 0, preds[n][-1], 'b', alpha=0.2)
        x.append(ydata.index[n + pred_window])

        y.append(preds[n][-1])
        y25.append(preds25[n][-1])
        y975.append(preds975[n][-1])
    plt.plot(x, y, 'r-', alpha=0.5, label='median prediction')
    # plt.plot(x, y25, 'b-', alpha=0.3)
    # plt.plot(x, y975, 'b-', alpha=0.3)
    plt.fill_between(x, np.array(y25), np.array(y975), color='b', alpha=0.3)

    plt.text(point, 0.6 * max_val, "Out of sample Predictions")
    plt.grid()
    plt.ylabel('Weekly cases')
    plt.title('Predictions for {}'.format(title))
    plt.xticks(rotation=70)
    plt.legend(loc=0)
    if save:
        if not os.path.exists('saved_models/' + path + '/' + STATE):
            os.mkdir('saved_models/' + path + '/' + STATE)

        plt.savefig(f'saved_models/{path}/{STATE}/qlgbm_{title}_ss.png', dpi=300)
    plt.show()
    return None


def qf_prediction(city, state, horizon, lookback):
    """
    Train model for a single city.
    :param city:
    :param state:
    :param horizon:
    :param lookback:
    :return:
    """
    with open(f'../analysis/clusters_{state}.pkl', 'rb') as fp:
        clusters = pickle.load(fp)
    data, group = get_cluster_data(city, clusters=clusters, data_types=DATA_TYPES, cols=PREDICTORS, doenca=DISEASE)

    target = f'casos_est_{city}'
    casos_est_columns = [f'casos_est_{i}' for i in group]
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
                                                        train_size=SPLIT, test_size=1 - SPLIT, shuffle=False)

    city_name = get_city_names([city, 0])[0][1]
    preds = np.empty((len(data_lag), horizon))
    preds25 = np.empty((len(data_lag), horizon))
    preds975 = np.empty((len(data_lag), horizon))
    metrics = pd.DataFrame(index=('mean_absolute_error', 'explained_variance_score',
                                  'mean_squared_error',
                                  'median_absolute_error', 'r2_score'))
    for d in range(1, horizon + 1):
        tgt = targets[d][:len(X_train)]
        tgtt = targets[d][len(X_train):]

        model25 = build_and_fit(X_train, target=tgt, quantile=0.025)
        model50 = build_and_fit(X_train, target=tgt, quantile=0.5)
        model975 = build_and_fit(X_train, target=tgt, quantile=0.975)
        dump(model50, f'saved_models/quantile_lgbm/{state}/{city}_city_model50_{d}W.joblib')
        dump(model25, f'saved_models/quantile_lgbm/{state}/{city}_city_model25_{d}W.joblib')
        dump(model975, f'saved_models/quantile_lgbm/{state}/{city}_city_model975_{d}W.joblib')
        pred25 = model25.predict(X_data[:len(targets[d])])
        pred = model50.predict(X_data[:len(targets[d])])
        pred975 = model975.predict(X_data[:len(targets[d])])

        dif = len(data_lag) - len(pred)
        if dif > 0:
            pred = list(pred) + ([np.nan] * dif)
            pred25 = list(pred25) + ([np.nan] * dif)
            pred975 = list(pred975) + ([np.nan] * dif)
        preds[:, (d - 1)] = pred
        preds25[:, (d - 1)] = pred25
        preds975[:, (d - 1)] = pred975

        pred_m = model50.predict(X_test[(d - 1):])
        print(d)
        metrics[d] = calculate_metrics(pred_m, tgtt)

    metrics.to_pickle(f'saved_models/quantile_lgbm/{state}/qlgbm_metrics_{city}.pkl')

    plot_prediction(preds, preds25, preds975, targets[1], city_name, len(X_train))

    return model50, preds, preds25, preds975, X_train, targets, data_lag


def qf_single_state_prediction(state, lookback, horizon, predictors):
    """
    QLGBM WITHOUT CLUSTER of SERIES
    :param state: 2-letter code for state
    :param lookback: number of steps of history to use
    :param horizon: number of weeks ahead to predict
    :param predictors: predictor variables
    :return:
    """

    if state == "CE":
        s = 'Ceará'
    else:
        s = state
    cities = list(get_cities_from_state(s))

    for city in cities:
        if os.path.isfile(f'/saved_models/quantile_lgbm_no_cluster/{state}/qlgbm_metrics_{city}.pkl'):
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
                                                            train_size=SPLIT, test_size=1 - SPLIT, shuffle=False)

        city_name = get_city_names([city, 0])[0][1]
        preds = np.empty((len(data_lag), horizon))
        metrics = pd.DataFrame(index=('mean_absolute_error', 'explained_variance_score',
                                      'mean_squared_error'
                                      'median_absolute_error', 'r2_score'))
        for d in range(1, horizon + 1):
            tgt = targets[d][:len(X_train)]
            tgtt = targets[d][len(X_train):]

            model = build_and_fit(X_train, target=tgt, quantile=0.5)
            pred = model.predict(X_data[:len(targets[d])])

            dif = len(data_lag) - len(pred)
            if dif > 0:
                pred = list(pred) + ([np.nan] * dif)
            preds[:, (d - 1)] = pred

            pred_m = model.predict(X_test[(d - 1):])
            metrics[d] = calculate_metrics(pred_m, tgtt)

        metrics.to_pickle('{}/{}/qlgbm_metrics_{}.pkl'.format('saved_models/quantile_lgbm_no_cluster', state, city))
        plot_prediction(preds, targets[1], city_name, len(X_train))
        # plt.show()


def state_forecast(state, horizon=4, lookback=4, plot=False):
    clusters = pd.read_pickle(f'../analysis/clusters_{state}.pkl')
    done = []
    forecasts = {}
    for clust in clusters:
        data_full, group = get_cluster_data(geocode=clust[0], clusters=clusters,
                                            data_types=DATA_TYPES, cols=PREDICTORS)
        data_lag = build_lagged_features(data_full, lookback)
        predindex = data_full.index.shift(4, freq='W')[-horizon:]
        for city in clust:
            if city in forecasts:
                continue
            target = 'casos_est_{}'.format(city)
            pred = np.empty(horizon)
            pred5 = np.empty(horizon)
            pred95 = np.empty(horizon)
            for d in range(1, horizon + 1):
                model5 = load(f'saved_models/quantile_lgbm/{state}/{city}_city_model5_{d}W.joblib')
                model50 = load(f'saved_models/quantile_lgbm/{state}/{city}_city_model50_{d}W.joblib')
                model95 = load(f'saved_models/quantile_lgbm/{state}/{city}_city_model95_{d}W.joblib')

                X_data = data_lag.iloc[-1:]
                pred[d-1] = model50.predict(X_data)
                pred5[d-1] = model5.predict(X_data)
                pred95[d-1] = model95.predict(X_data)

            city_name = get_city_names([city, 0])[0][1]
            forecasts[city] = (data_lag[target].iloc[-lookback:], pred, pred5, pred95, predindex, city_name)
            if plot:
                plot_forecast(*forecasts[city])

    return forecasts


def plot_forecast(data, pred, pred5, pred95, predindex, city_name):
    fig, [ax, ax1] = plt.subplots(2, 1)
    data.plot(ax=ax, label='cases')
    ax1.plot(predindex, pred, 'r-*', label='median')
    ax1.fill_between(predindex, pred5, pred95, color='b', alpha=0.3)
    ax.set_title(f"Forecast for {city_name}")
    ax.get_xaxis().Tick(labelrotation=70)
    ax1.get_xaxis().Tick(labelrotation=70)
    ax.legend(loc=0)
    ax1.legend(loc=0)
    ax.grid()
    ax1.grid()
    plt.show()


def qf_state_prediction(state, lookback, horizon, predictors):
    """
    RQF prediction based on cluster of cities
    :param state:
    :param lookback:
    :param horizon:
    :param predictors:
    :return:
    """
    clusters = pd.read_pickle(f'../analysis/clusters_{state}.pkl')

    for cluster in clusters:
        data_full, group = get_cluster_data(geocode=cluster[0], clusters=clusters,
                                            data_types=DATA_TYPES, cols=PREDICTORS)
        for city in cluster:
            if os.path.isfile(
                    f'saved_models/quantile_lgbm/{state}/qlgbm_metrics_{city}.pkl'):
                print('done')
                continue

            target = 'casos_est_{}'.format(city)
            casos_est_columns = ['casos_est_{}'.format(i) for i in group]
            # casos_columns = ['casos_{}'.format(i) for i in group]

            # data = data_full.drop(casos_columns, axis=1)
            data_lag = build_lagged_features(data_full, lookback)
            data_lag.dropna()
            targets = {}
            for d in range(1, horizon + 1):
                if d == 1:
                    targets[d] = data_lag[target].shift(-(d - 1))
                else:
                    targets[d] = data_lag[target].shift(-(d - 1))[:-(d - 1)]

            X_data = data_lag  # .drop(casos_est_columns, axis=1)
            if len(X_data) == 0:
                print(f"No data available for {city}, {state}")
                continue
            X_train, X_test, y_train, y_test = train_test_split(X_data, data_lag[target],
                                                                train_size=SPLIT, test_size=1 - SPLIT, shuffle=False)

            city_name = get_city_names([city, 0])[0][1]
            preds = np.empty((len(data_lag), horizon))
            preds5 = np.empty((len(data_lag), horizon))
            preds95 = np.empty((len(data_lag), horizon))
            metrics = pd.DataFrame(index=('mean_absolute_error', 'explained_variance_score',
                                          'mean_squared_error', 'median_absolute_error', 'r2_score'))
            for d in range(1, horizon + 1):
                tgt = targets[d][:len(X_train)]
                tgtt = targets[d][len(X_train):]

                model5 = build_and_fit(X_train, target=tgt, quantile=0.05)
                model50 = build_and_fit(X_train, target=tgt, quantile=0.5)
                model95 = build_and_fit(X_train, target=tgt, quantile=0.95)

                dump(model5, f'saved_models/quantile_lgbm/{state}/{city}_city_model5_{d}W.joblib')
                dump(model50, f'saved_models/quantile_lgbm/{state}/{city}_city_model50_{d}W.joblib')
                dump(model95, f'saved_models/quantile_lgbm/{state}/{city}_city_model95_{d}W.joblib')

                pred = model50.predict(X_data[:len(targets[d])])
                pred5 = model5.predict(X_data[:len(targets[d])])
                pred95 = model95.predict(X_data[:len(targets[d])])

                dif = len(data_lag) - len(pred)
                if dif > 0:
                    pred = list(pred) + ([np.nan] * dif)
                    pred5 = list(pred5) + ([np.nan] * dif)
                    pred95 = list(pred95) + ([np.nan] * dif)
                preds[:, (d - 1)] = pred
                preds5[:, (d - 1)] = pred5
                preds95[:, (d - 1)] = pred95

                pred_m = model50.predict(X_test[(d - 1):])
                metrics[d] = calculate_metrics(pred_m, tgtt)

            metrics.to_pickle(f'saved_models/quantile_lgbm/{state}/qlgbm_metrics_{city}.pkl')
            dump(model50, f'saved_models/quantile_lgbm/{state}/{city}_state_model50.joblib')
            dump(model5, f'saved_models/quantile_lgbm/{state}/{city}_state_model5.joblib')
            dump(model95, f'saved_models/quantile_lgbm/{state}/{city}_state_model50.joblib')
            plot_prediction(preds, preds5, preds95, targets[1], city_name, len(X_train))
            # plt.show()


if __name__ == "__main__":
    # target = 'casos_est_{}'.format(CITY)
    # preds = qf_prediction(CITY, STATE, target, PREDICTION_WINDOW, LOOK_BACK)
    # for STATE in ['RJ', 'PR', 'CE']:
    # qf_state_prediction(STATE, LOOK_BACK, PREDICTION_WINDOW, PREDICTORS)

    # qf_single_state_prediction(STATE, LOOK_BACK, PREDICTION_WINDOW, PREDICTORS)

    # model, preds, preds25, preds975, X_train, targets, data_lag = qf_prediction(CITY, STATE,
    #                                                                             horizon=PREDICTION_WINDOW,
    #                                                                             lookback=LOOK_BACK)
    # print(model.feature_importances_)

    frcsts = state_forecast(STATE, PREDICTION_WINDOW, LOOK_BACK, plot=True)
