import numpy as np
import pandas as pd
from matplotlib import pyplot as P
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils.visualize_util import plot
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets
from sklearn.preprocessing import normalize, LabelEncoder
from time import time
from infodenguepredict.data.infodengue import get_alerta_table, get_temperature_data, get_tweet_data, build_multicity_dataset

HIDDEN = 256
TIME_WINDOW = 12
BATCH_SIZE = 1


def split_data(df, n_weeks=12, ratio=0.8, predict_n=5, Y_column=0):
    """
    Split the data into training and test sets
    Keras expects the input tensor to have a shape of (nb_samples, timesteps, features).
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
    Y_train = train[n_weeks:, -predict_n:, Y_column]
    X_test = test[:-n_weeks, :n_weeks, :]
    Y_test = test[n_weeks:, -predict_n:, Y_column]

    return X_train, Y_train, X_test, Y_test


def build_model(hidden, features, time_window=10, batch_size=1):
    """
    Builds and returns the LSTM model with the parameters given
    :param hidden: number of hidden nodes
    :param features: number of variables in the example table
    :param time_window: Number of time-steps to look back before predicting
    :param batch_size: batch size for batch training
    :return:
    """
    model = Sequential()

    # model.add(LSTM(hidden, input_shape=(time_window, features), stateful=True,
    #                batch_input_shape=(batch_size, time_window, features), return_sequences=True))
    # model.add(Dropout(0.2))

    model.add(LSTM(hidden, input_shape=(time_window, features), stateful=True,
                   batch_input_shape=(batch_size, time_window, features)))
    model.add(Dropout(0.2))

    model.add(Dense(prediction_window))  # five time-step ahead prediction
    model.add(Activation("linear"))

    start = time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time() - start)
    plot(model, to_file='model.png')
    return model


def train(model, X_train, Y_train, batch_size=1, epochs=20, overwrite=True):
    hist = model.fit(X_train, Y_train,
                     batch_size=batch_size, nb_epoch=epochs, validation_split=0.05, verbose=1)
    model.save_weights('trained_model.h5', overwrite=overwrite)
    return hist


def plot_training_history(hist):
    """
    Plot the Loss series from training the model
    :param hist: Training history object returned by "model.fit()"
    """
    df_vloss = pd.DataFrame(hist.history['val_loss'], columns=['val_loss'])
    df_loss = pd.DataFrame(hist.history['loss'], columns=['loss'])
    ax = df_vloss.plot(logy=True);
    df_loss.plot(ax=ax, grid=True, logy=True);
    P.savefig("LSTM_training_history.png")


def get_example_table(geocode=None):
    """
    Fetch the data from the database, filters out useless variables
    :return: pandas dataframe
    """
    raw_df = get_alerta_table(geocode)
    filtered_df  = raw_df[['SE', 'casos_est', 'casos_est_min', 'casos_est_max',
       'casos', 'municipio_geocodigo', 'p_rt1', 'p_inc100k', 'nivel']]
    filtered_df['SE'] = [int(str(x)[-2:]) for x in filtered_df.SE]

    return filtered_df


def get_complete_table(geocode=None):
    """
    Extends Example table with temperature, humidity atmospheric pressure and Tweets
    :param geocode:
    :return:
    """
    df = get_example_table(geocode=geocode)
    T = get_temperature_data(geocode)
    Tw = get_tweet_data(municipio=geocode)
    Tw.pop('Municipio_geocodigo')
    Tw.pop('CID10_codigo')
    complete = df.join(T).join(Tw).dropna()
    return complete



def normalize_data(df):
    """
    Normalize features in the example table
    :param df:
    :return:
    """
    if 'municipio_geocodigo' in df.columns:
        df.pop('municipio_geocodigo')

    for col in df.columns:
        if col.startswith('nivel'):
            # print(col)
            le = LabelEncoder()
            le.fit(df[col])
            df[col] = le.transform(df[col])

    norm = normalize(df, norm='l2', axis=0)
    df_norm = pd.DataFrame(norm, columns=df.columns)

    return df_norm

def plot_predicted_vs_data(model, Xdata, Ydata, label, pred_window):
    P.clf()
    predicted = model.predict(Xdata, batch_size=BATCH_SIZE, verbose=1)
    df_predicted = pd.DataFrame(predicted).T
    for n in range(df_predicted.shape[1]):
        P.plot(range(n, n + pred_window), pd.DataFrame(Ydata.T)[n], 'y-')
        P.plot(range(n, n + pred_window), df_predicted[n], 'g:o')
    P.grid()
    P. title(label)
    P.xlabel('weeks')
    P.ylabel('normalized incidence')
    P.legend([label, 'predicted'])
    P.savefig("lstm_{}.png".format(label))


def loss_and_metrics(model, Xtest, Ytest):
    print(model.evaluate(Xtest, Ytest, batch_size=1))


if __name__ == "__main__":
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
                                                  n_weeks=TIME_WINDOW, ratio=.7,
                                                  predict_n=prediction_window, Y_column=target_col)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    model = build_model(HIDDEN, X_train.shape[2], TIME_WINDOW, BATCH_SIZE)
    history = train(model, X_train, Y_train, batch_size=1, epochs=30)
    model.save('lstm_model')
    ## plotting results
    print(model.summary())
    loss_and_metrics(model, X_test, Y_test)
    plot_training_history(history)
    plot_predicted_vs_data(model, X_train, Y_train, label='In Sample', pred_window=prediction_window)
    plot_predicted_vs_data(model, X_test, Y_test, label='Out of Sample', pred_window=prediction_window)
    P.show()
