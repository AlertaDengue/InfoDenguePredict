import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import pickle
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
        res = pd.concat([dt.shift(i) for i in the_range], axis=1)
        res.columns = ['lag_%d' % i for i in the_range]
    else:
        print('Only works for DataFrame or Series')
        return None
    if dropna:
        return res.dropna()
    else:
        return res


def rolling_forecasts(data, window=12, horizon=1, target=None):
    model = build_model()
    ldf = build_lagged_features(data, window)
    # print(ldf.head())
    for i in range(0, ldf.shape[0] - window):
        model.fit(ldf.values[i:i + window, :], ldf[target].values[i:i+window])
    plot_prediction(ldf.values, ldf[target].values, model)
    return model



def plot_prediction(Xdata, ydata, model):
    preds = model.predict(Xdata)
    plt.plot(ydata, label='Data')
    plt.plot(preds, label='RF')
    plt.legend(loc=0)
    plt.show()


if __name__ == "__main__":
    prediction_window = 5  # weeks
    city = 3304557
    with open('../analysis/clusters_{}.pkl'.format(STATE), 'rb') as fp:
        clusters = pickle.load(fp)
    data, group = get_cluster_data(city, clusters=clusters, data_types=DATA_TYPES, cols=PREDICTORS)
    # print(data.head())
    # data.casos_est.plot()
    model = rolling_forecasts(data, target='casos_{}'.format(city))

    print(model.feature_importances_)

