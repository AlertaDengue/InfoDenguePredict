import numpy as np
import pandas as pd
import scipy.stats as ss

from matplotlib import pyplot as P

from infodenguepredict.data.infodengue import get_alerta_table, get_cluster_data, random_data, get_city_names
from infodenguepredict.models.deeplearning.lstm import single_prediction
from infodenguepredict.predict_settings import *


def create_px(row):
    res = ss.probplot([row['preds']], dist = row['dists'])
    return res[0][1][0]


def create_qy(row):
    res = ss.probplot([row['real']], dist = row['dists'])
    return res[0][1][0]


def create_rvs(data, dist='expon'):
    Q = data.groupby('SE')
    dists = []
    for group in Q.groups:
        serie = Q.get_group(group).casos_est.values

        if dist == 'expon':
            par = ss.expon.fit(serie)
            rv = ss.expon(par[0], par[1])

        if dist == 'gamma':
            par = ss.gamma.fit(serie)
            rv = ss.gamma(par[0], par[1], par[2])

        if dist == 'lognorm':
            par = ss.lognorm.fit(serie)
            rv = ss.lognorm(par[0], par[1], par[2])

        dists.append(rv)

    rvs = pd.DataFrame({'dists': dists, 'SE': list(Q.groups.keys())})
    return rvs


def qqplot(predicted, real, city, state, look_back, all_predict_n=False):
    """
    Plot QQPlot for prediction values
    :param predicted: Predicted matrix
    :param real: Array of target_col values used in the prediction
    :param city: Geocode of the target city predicted
    :param state: State containing the city
    :param look_back: Look-back time window length used by the model
    :param all_predict_n: If True, plot the qqplot for every value predicted
    :return:
    """
    data_full = get_alerta_table(city, 'RJ')
    data_full = data_full[['casos_est', 'SE']].reset_index(drop=True)
    data_full.SE = [str(i)[-2:] for i in data_full.SE]

    rvs = create_rvs(data_full)

    if all_predict_n:
        fig, axs = plt.subplots(4, 1, figsize=[6, 20])
        weeks = list(range(predicted.shape[1]))
    else:
        fig, axs = plt.subplots(1, 1, figsize=[6, 5])
        weeks = [0]

    for week in weeks:
        data = data_full.iloc[look_back + week:len(real) + look_back + week]
        data['preds'] = predicted[:, week]
        data['real'] = real[:, week]

        data = data.merge(rvs, on='SE', how='left')
        data['p_preds'] = data.apply(create_px, axis=1)
        data['q_real'] = data.apply(create_qy, axis=1)

        train = int(len(real) * 0.7)
        preds_max = int(data['p_preds'].max())

        axs.plot(list(range(preds_max)), list(range(preds_max)), color='black')
        data[train:].plot(x='p_preds', y='q_real', kind='scatter', ax=axs, color='b', legend=True)
        data[:train].plot(x='p_preds', y='q_real', kind='scatter', ax=axs, color='r', legend=True)
        # P.show()
    return data


if __name__ == "__main__":
    predicted, X_test, Y_test, Y_train, factor = single_prediction(CITY, STATE, PREDICTORS, predict_n=PREDICTION_WINDOW,
                                                                   look_back=LOOK_BACK,
                                                                   hidden=HIDDEN, epochs=50)

    real = np.concatenate((Y_train*factor, Y_test*factor))
    preds = predicted*factor
    df_data = qqplot(predicted, real, CITY, STATE, LOOK_BACK)