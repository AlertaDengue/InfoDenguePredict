import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor
import pickle
import forestci as fci
from datetime import datetime
import matplotlib.pyplot as plt
from infodenguepredict.data.infodengue import get_cluster_data
from infodenguepredict.predict_settings import PREDICTORS, DATA_TYPES, STATE


def build_model(**kwargs):
    model = RandomForestRegressor(max_depth=None, random_state=0, n_jobs=-1,
                                  n_estimators=1000,
                                  warm_start=True)
    # model = ExtraTreesRegressor(max_depth=None, random_state=0, n_jobs=-1,
    #                               n_estimators=1000,
    #                               warm_start=True)

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


def rolling_forecasts(data, target, window=12, horizon=1):
    """
    Fits the rolling forecast model
    :param data: feature Dataframe
    :param window: lookback window
    :param horizon: forecast horizon
    :param target: variable to be forecasted
    :return:
    """
    model = build_model()
    model.fit(data.values, target)
    # for i in range(0, ldf.shape[0] - window):
    #     model.fit(ldf.values[i:i + window, :], ldf['target'].values[i:i + window])

    return model


def plot_prediction(Xdata, ydata, model, title):
    plt.figure()
    preds = model.predict(Xdata)
    plt.plot(ydata, alpha=0.3, label='Data')
    plt.plot(preds, ':', label='RandomForest')
    plt.legend(loc=0)
    plt.title(title)
    plt.savefig('RandomForest{}_{}.png'.format(city, title))
    return preds

def confidence_interval(model, Xtrain, Xtest):
    inbag = fci.calc_inbag(X_train.shape[0], model)
    ci = fci.random_forest_error(model, Xtrain.values, Xtest.values, inbag=inbag)
    return ci

if __name__ == "__main__":
    lookback = 12
    horizon = 5  # weeks
    city = 3304557
    target = 'casos_{}'.format(city)
    with open('../analysis/clusters_{}.pkl'.format(STATE), 'rb') as fp:
        clusters = pickle.load(fp)
    data, group = get_cluster_data(city, clusters=clusters, data_types=DATA_TYPES, cols=PREDICTORS)

    data_lag = build_lagged_features(data, lookback)
    data_lag.dropna()
    targets = {}
    for d in range(1, horizon + 1):
        targets[d] = data_lag[target].shift(-d)[:-horizon]

    X_train, X_test, y_train, y_test = train_test_split(data_lag, data_lag[target],
                                                        train_size=0.75, test_size=0.25, shuffle=False)
    X_test = X_test.iloc[:-horizon]

    X_train[target].plot()
    targets[2].plot(label='target')
    plt.legend(loc=0)
    plt.show()

    for d in range(1, horizon + 1):
        tgt = targets[d][:len(X_train)]
        tgtt = targets[d][len(X_train):]
        model = rolling_forecasts(X_train, target=tgt, horizon=horizon)

        ci = confidence_interval(model, X_train, X_test)
        print(ci)
        
        plot_prediction(X_test.values, tgtt.values, model, 'Out_of_Sample_{}'.format(d))
        plt.show()
        break
