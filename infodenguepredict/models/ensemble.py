"""
Use Tpot regressor to test varius models on the problem
Ensemble model
"""

from tpot import TPOTRegressor
import pickle
from infodenguepredict.data.infodengue import get_cluster_data
from infodenguepredict.predict_settings import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd


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


def rolling_forecasts(data, target):
    """
    Fits the rolling forecast model
    :param data: feature Dataframe
    :param window: lookback window
    :param horizon: forecast horizon
    :param target: variable to be forecasted
    :return:
    """
    model = TPOTRegressor(generations=5, population_size=50, verbosity=2)
    model.fit(data.values, target)
    # for i in range(0, ldf.shape[0] - window):
    #     model.fit(ldf.values[i:i + window, :], ldf['target'].values[i:i + window])

    return model

def plot_prediction(Xdata, ydata, model, title, shift, horizon=None):
    plt.figure()
    preds = model.predict(Xdata.values)
    # pred_in = pred[:-(len(ydata)+shift)]
    # pred_out = pred[-(len(ydata)+shift):-shift]
    plt.plot(ydata, alpha=0.3, label='Data')

    preds_series = pd.Series(data=preds, index=list(ydata.index))

    plt.plot(preds_series, ':', label='RandomForest')

    plt.legend(loc=0)
    plt.title(title)
    plt.savefig('{}/TPOT{}.png'.format(FIG_PATH, title), dpi=300)

    return preds_series

# def plot_prediction(Xdata, ydata, model, title):
#     plt.figure()
#     preds = model.predict(Xdata)
#     plt.plot(ydata, alpha=0.3, label='Data')
#     plt.plot(preds, ':', label='Tpot')n)
#     lX_test.dropna(inplace=True)
#     lX_train[target].plot()
#     lX_train.target.plot()
#     plt.legend(loc=0)
#     plt.show()
#     plt.legend(loc=0)
#     plt.title(title)
#     plt.savefig('TPOT{}_{}.png'.format(city, title))


def ensemble_tpot(city, state, target, horizon, lookback):
    with open('../analysis/clusters_{}.pkl'.format(state), 'rb') as fp:
        clusters = pickle.load(fp)
        data, group = get_cluster_data(city, clusters=clusters, data_types=DATA_TYPES, cols=PREDICTORS)

    casos_est_columns = ['casos_est_{}'.format(i) for i in group]
    casos_columns = ['casos_{}'.format(i) for i in group]

    data = data.drop(casos_columns, axis=1)
    data_lag = build_lagged_features(data, lookback)
    data_lag.dropna()

    X_data = data_lag.drop(casos_est_columns, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_data, data_lag[target],
                                                        train_size=0.7, test_size=0.3, shuffle=False)

    tgt_full = data_lag[target].shift(-(horizon - 1))[:-(horizon - 1)]
    tgt = tgt_full[:len(X_train)]
    tgtt = tgt_full[len(X_train):]

    model = TPOTRegressor(generations=20, population_size=100, verbosity=2)
    model.fit(X_train, target=tgt)
    model.export('tpot_{}_pipeline.py'.format(city))
    print(model.score(X_test[:len(tgtt)], tgtt))

    pred = plot_prediction(X_data[:len(tgt_full)], tgt_full, model, 'Out_of_Sample_{}_{}'.format(horizon, city), horizon)
    plt.show()
    return pred


if __name__ == "__main__":
    target = 'casos_est_{}'.format(CITY)

    preds = ensemble_tpot(CITY, STATE, target=target, horizon=4, lookback=4)
