"""
This script performs LSTM cross-disease predictions for a single city only.
"""
import numpy as np
import pandas as pd
import pickle
import math
import os
import shap

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Tesla K40

from matplotlib import pyplot as P
import keras
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard, EarlyStopping
from keras import backend as K
from sklearn.metrics import *
import matplotlib.pyplot as plt

from time import time
from infodenguepredict.data.infodengue import (
    combined_data,
    get_cluster_data,
    random_data,
    get_city_names,
)
from infodenguepredict.models.deeplearning.preprocessing import (
    split_data,
    normalize_data,
)
from infodenguepredict.predict_settings import *


def plot_prediction(pred, pred25, pred975, x, ydata, factor, horizon, title, path='LSTM', save=True, doenca='chik'):
    plt.clf()
    plt.plot(ydata, 'k-', label='data')
    x = x[7:]
    pred['date'] = pd.to_datetime(x)
    pred.set_index('date', inplace=True)
    pred25['date'] = pd.to_datetime(x)
    pred25.set_index('date', inplace=True)
    pred975['date'] = pd.to_datetime(x)
    pred975.set_index('date', inplace=True)
    # x = ydata.index.shift(horizon, freq='W')
    plt.plot(x, ydata[:, -1] * factor, 'k-', alpha=0.7, label='data')
    plt.plot(x, pred[3].values * factor, 'r-', alpha=0.5, label='median')
    plt.fill_between(x, pred25[3].values * factor,
                   pred975[3].values * factor,
                   color='b', alpha=0.3)


    plt.grid()
    plt.ylabel('Weekly cases')
    plt.title('LSTM {} cross-predictions for {}'.format(doenca, title))
    # plt.xticks(rotation=70)
    plt.legend(loc=0)
    if save:
        if not os.path.exists('../saved_models/' + path + '/' + STATE):
            os.mkdir('../saved_models/' + path + '/' + STATE)

        plt.savefig('../saved_models/{}/{}/lstm_{}_cross_{}_.png'.format(path, STATE, doenca, title), dpi=300)
    plt.show()
    return None


def single_prediction(city, state, predictors, predict_n, look_back, hidden, epochs, predict=True, doenca='chick'):
    """
    Fit an LSTM model to generate predictions for a city, Using its cluster as regressors.
    :param city: geocode of the target city
    :param state: State containing the city
    :param predict_n: How many weeks ahead to predict
    :param look_back: Look-back time window length used by the model
    :param hidden: Number of hidden layers in each LSTM unit
    :param epochs: Number of epochs of training
    :param predict: Only generate predictions
    :param random: If the model should be trained on a random selection of ten cities of the same state.
    :return:
    """

    with open("../../analysis/clusters_{}.pkl".format(state), "rb") as fp:
        clusters = pickle.load(fp)
    data, group = get_cluster_data(
        geocode=city, clusters=clusters, data_types=DATA_TYPES, cols=predictors, doenca=doenca
    )
    data = data['2016-01-01':]
    x = data.index.shift(predict_n, freq='W')
    x = [i.date() for i in x]
    indice = list(data.index)
    indice = [i.date() for i in indice]

    city_name = get_city_names([city, 0])[0][1]
    if predict:
        ratio = 1
    else:
        ratio = 0.7

    if cluster:
        target_col = list(data.columns).index("casos_est_{}".format(city))
    else:
        target_col = list(data.columns).index("casos_est")
    norm_data, max_features = normalize_data(data)
    factor = max_features[target_col]
    ## split test and train
    X_train, Y_train, X_test, Y_test = split_data(
        norm_data,
        look_back=look_back,
        ratio=ratio,
        predict_n=predict_n,
        Y_column=target_col,
    )

    model = load_model("../saved_models/LSTM/{}/lstm_{}_epochs_{}.h5".format(state, city, epochs))
    predicted = np.stack([model.predict(X_train, batch_size=1, verbose=1) for i in range(100)], axis=2)

    df_predicted = pd.DataFrame(np.percentile(predicted, 50, axis=2))
    df_predicted25 = pd.DataFrame(np.percentile(predicted, 2.5, axis=2))
    df_predicted975 = pd.DataFrame(np.percentile(predicted, 97.5, axis=2))

    plot_prediction(
        pred=df_predicted,
        pred25=df_predicted25,
        pred975=df_predicted975,
        x=x,
        ydata=Y_train,
        factor=factor,
        horizon=predict_n,
        title="{}".format(city_name),
        doenca=doenca
    )

    return predicted, indice, X_test, Y_test, Y_train, factor


if __name__ == "__main__":
    # K.set_epsilon(1e-5)

    predicted, indice, X_test, Y_test, Y_train, factor = single_prediction(
        CITY,
        STATE,
        PREDICTORS,
        predict_n=PREDICTION_WINDOW,
        look_back=LOOK_BACK,
        hidden=HIDDEN,
        epochs=EPOCHS,
        doenca='chik'
    )
