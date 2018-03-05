"""
Created on 27/01/17
by fccoelho
license: GPL V3 or Later
adapted from this example:
http://machinelearningmastery.com/time-series-prediction-with-deep-learning-in-python-with-keras/
"""

import numpy as np
import pandas as pd
from time import time
from matplotlib import pyplot as P
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, classification_report
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils.visualize_util import plot
from infodenguepredict.models.deeplearning.preprocessing import split_data, normalize_data
from infodenguepredict.data.infodengue import get_alerta_table, get_temperature_data, get_tweet_data, \
    build_multicity_dataset


def build_model(hidden, features, look_back=10, batch_size=1):
    """
    Builds and returns the MLP model with the parameters given
    :param hidden: number of hidden nodes
    :param features: number of variables in the example table
    :param look_back: Number of time-steps to look back before predicting
    :param batch_size: batch size for batch training
    :return:
    """
    model = Sequential()

    model.add(Dense(hidden, input_shape=(look_back, features)))
    # model.add(Dropout(0.2))

    model.add(Dense((prediction_window), activation='relu'))  # five time-step ahead prediction

    start = time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time() - start)
    plot(model, to_file='model.png')
    return model


def train(model, X_train, Y_train, batch_size=1, epochs=20, overwrite=True):
    hist = model.fit(X_train, Y_train,
                     batch_size=batch_size, nb_epoch=epochs, validation_split=0.05, verbose=1)
    model.save_weights('trained_lstm_model.h5', overwrite=overwrite)
    return hist


if __name__ == "__main__":
    HIDDEN = 256
    TIME_WINDOW = 12
    BATCH_SIZE = 1
    prediction_window = 2  # weeks
    # data = get_example_table(3304557) #Nova Iguaçu: 3303500
    # data = get_complete_table(3304557)
    data = build_multicity_dataset('RJ')
    print(data.shape)
    target_col = list(data.columns).index('casos_est_3303500')
    time_index = data.index
    norm_data = normalize_data(data)
    print(norm_data.columns, norm_data.shape)
    # norm_data.casos_est.plot()
    # P.show()
    X_train, Y_train, X_test, Y_test = split_data(norm_data,
                                                  look_back=TIME_WINDOW, ratio=.7,
                                                  predict_n=prediction_window, Y_column=target_col)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    model = build_model(HIDDEN, X_train.shape[2], TIME_WINDOW, BATCH_SIZE)
    history = train(model, X_train, Y_train, batch_size=1, epochs=30)
    model.save('mlp_model')
