import numpy as np
import pandas as pd
import pylab as P
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets
from time import time

HIDDEN = 32
NFEATURES = 90
TIME_WINDOW = 12
BATCH_SIZE = 1

@property
def Model():
    return build_model(HIDDEN, NFEATURES, TIME_WINDOW, BATCH_SIZE)


def split_data(df, n_weeks=12, ratio=0.8, predict_n=5, Y_column=0):
    """
    Split the data into training and test sets

    :param df: Pandas dataframe with the data.
    :param n_weeks: Number of weeks to look back before predicting
    :param ratio: fraction of total samples to use for training
    :param predict_n: number of weeks to predict
    :param Y_column: Column to predict
    :return:
    """
    df = np.nan_to_num(df.values).astype("float32")
    # n_ts is the number of training samples also number of training sets
    # since windows have an overlap of n-1
    n_ts = df.shape[0] - n_weeks - predict_n
    data = np.empty((n_ts, n_weeks + predict_n, df.shape[1]))
    for i in range(n_ts - predict_n):
        #         print(i, df[i: n_weeks+i+predict_n,0])
        data[i, :, :] = df[i: n_weeks + i + predict_n, :]
    train_size = int(n_ts * ratio)
    print(train_size)
    train = data[:train_size, :, :]
    test = data[train_size:, :, :]
    #     np.random.shuffle(train)
    # We are predicting only column 0
    X_train = train[:-n_weeks, :n_weeks, :]
    Y_train = train[n_weeks:, -predict_n:, 0]
    X_test = test[:-n_weeks, :n_weeks, :]
    Y_test = test[n_weeks:, -predict_n:, 0]

    return X_train, Y_train, X_test, Y_test


def build_model(hidden, features, time_window=10, batch_size=1):
    """
    Builds and returns the LSTM model with the parameters given
    :param hidden:
    :param features:
    :param time_window:
    :param batch_size:
    :return:
    """
    model = Sequential()

    model.add(LSTM(hidden, input_shape=(time_window, features), stateful=True,
                   batch_input_shape=(batch_size, time_window, features), return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(hidden, input_shape=(time_window, features), stateful=True,
                   batch_input_shape=(batch_size, time_window, features)))
    model.add(Dropout(0.2))

    model.add(Dense(prediction_window))  # five time-step ahead prediction
    model.add(Activation("linear"))

    start = time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time() - start)
    return model


def train(model, overwrite=True):
    hist = model.fit(X_train, Y_train,
                     batch_size=bs, nb_epoch=20, validation_split=0.05, verbose=1)
    model.save_weights('trained_model.h5', overwrite=overwrite)
    return hist


def plot_training_history(hist):
    """
    Plot the Loss series from training the model
    :param hist: Training history object returned by "model.fit()"
    """
    df_vloss = pd.DataFrame(hist.history['val_loss'], columns=['val_loss'])
    df_loss = pd.DataFrame(hist.history['loss'], columns=['loss'])
    ax = df_vloss.plot();
    df_loss.plot(ax=ax, grid=True);
    P.savefig("training_history.png")


if __name__ == "__main__":
    prediction_window = 5  # weeks
    X_train, Y_train, X_test, Y_test = split_data(features_norm_df,
                                                  n_weeks=12, ratio=.8,
                                                  predict_n=prediction_window)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    model = build_model(HIDDEN, NFEATURES, TIME_WINDOW, BATCH_SIZE)
    history = train(model)

