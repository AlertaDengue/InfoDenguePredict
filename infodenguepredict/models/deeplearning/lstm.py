import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as P
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard
from hyperas.distributions import uniform, choice
from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe

from time import time
from infodenguepredict.data.infodengue import get_alerta_table, get_temperature_data, get_tweet_data, get_cluster_data
from infodenguepredict.models.deeplearning.preprocessing import split_data, normalize_data

HIDDEN = 4
TIME_WINDOW = 2
BATCH_SIZE = 1


def optimize_model(x_train, y_train, x_test, y_test, features, TIME_WINDOW):

    model = Sequential()

    model.add(LSTM({{choice([4, 8, 16])}}, input_shape=(TIME_WINDOW, features), stateful=True,
                   batch_input_shape=(1, TIME_WINDOW, features), return_sequences=True))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(LSTM({{choice([4, 8, 16])}}, input_shape=(TIME_WINDOW, features), stateful=True,
                   batch_input_shape=(1, TIME_WINDOW, features), return_sequences=True))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(LSTM({{choice([4, 8, 16])}}, input_shape=(TIME_WINDOW, features), stateful=True,
                   batch_input_shape=(1, TIME_WINDOW, features)))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Dense(prediction_window, activation='relu'))

    start = time()
    model.compile(loss="mse", optimizer="rmsprop")
    model.fit(x_train, y_train,
              batch_size=1,
              nb_epoch=1,
              validation_split=0.05,
              verbose=0,
              )
    loss = model.evaluate(x_test, y_test, batch_size=1)
    return {'loss': loss, 'status': STATUS_OK, 'model': model}

def build_model(hidden, features, look_back=10, batch_size=1):
    """
    Builds and returns the LSTM model with the parameters given
    :param hidden: number of hidden nodes
    :param features: number of variables in the example table
    :param look_back: Number of time-steps to look back before predicting
    :param batch_size: batch size for batch training
    :return:
    """
    model = Sequential()

    model.add(LSTM(hidden, input_shape=(look_back, features), stateful=True,
                   batch_input_shape=(batch_size, look_back, features),
                   return_sequences=True,
                   dropout=0.1,
                   recurrent_dropout=0.1
                   ))
    model.add(LSTM(hidden, input_shape=(look_back, features), stateful=True,
                   batch_input_shape=(batch_size, look_back, features),
                   return_sequences=True,
                   dropout=0.1,
                   recurrent_dropout=0.1
                   ))

    model.add(LSTM(hidden, input_shape=(look_back, features), stateful=True,
                   batch_input_shape=(batch_size, look_back, features),
                   dropout=0.1,
                   recurrent_dropout=0.1
                   ))

    model.add(Dense(prediction_window, activation='relu'))

    start = time()
    model.compile(loss="poisson", optimizer="rmsprop")
    print("Compilation Time : ", time() - start)
    plot_model(model, to_file='LSTM_model.png')
    return model


def train(model, X_train, Y_train, batch_size=1, epochs=100, overwrite=True):
    TB_callback = TensorBoard(log_dir='./tensorboard',
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True,
                              # embeddings_freq=10
                              )

    hist = model.fit(X_train, Y_train,
                     batch_size=batch_size,
                     nb_epoch=epochs,
                     validation_split=0.05,
                     verbose=1,
                     callbacks=[TB_callback])
    model.save_weights('trained_lstm_model.h5', overwrite=overwrite)
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



def plot_predicted_vs_data(model, Xdata, Ydata, label, pred_window):
    P.clf()
    predicted = model.predict(Xdata, batch_size=BATCH_SIZE, verbose=1)
    df_predicted = pd.DataFrame(predicted).T
    for n in range(df_predicted.shape[1]):
        P.plot(range(n, n + pred_window), pd.DataFrame(Ydata.T)[n], 'k-')
        P.plot(range(n, n + pred_window), df_predicted[n], 'g:o', alpha=0.5)
    P.grid()
    P. title(label)
    P.xlabel('weeks')
    P.ylabel('normalized incidence')
    P.legend([label, 'predicted'])
    P.savefig("lstm_{}.png".format(label))


def loss_and_metrics(model, Xtest, Ytest):
    print(model.evaluate(Xtest, Ytest, batch_size=1))


if __name__ == "__main__":
    prediction_window = 3  # weeks
    city = 3304557
    state = 'RJ'

    with open('../clusters_{}.pkl'.format(state), 'rb') as fp:
        clusters = pickle.load(fp)

    # data = get_example_table(3304557) #Nova Iguaçu: 3303500
    # data = get_complete_table(3304557)
    # data = build_multicity_dataset('RJ')
    data = get_cluster_data(city, clusters)
    # print(data.shape)

    target_col = list(data.columns).index('casos_{}'.format(city))
    # print(target_col)
    time_index = data.index
    norm_data = normalize_data(data)
    # print(norm_data.columns, norm_data.shape)
    # norm_data.casos_3304557.plot()
    # P.show()
    X_train, Y_train, X_test, Y_test = split_data(norm_data,
                                                  look_back=TIME_WINDOW, ratio=.7,
                                                  predict_n=prediction_window, Y_column=target_col)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    ## Optimize Hyperparameters
    #
    # def get_data():
    #     return X_train, Y_train, X_test, Y_test, X_train.shape[2], TIME_WINDOW
    #
    # best_run, best_model = optim.minimize(model=optimize_model,
    #                                       data=get_data,
    #                                       algo=tpe.suggest,
    #                                       max_evals=5,
    #                                       trials=Trials())
    ## Run model
    model = build_model(HIDDEN, X_train.shape[2], TIME_WINDOW, BATCH_SIZE)
    history = train(model, X_train, Y_train, batch_size=1, epochs=50)
    model.save('lstm_model')
    ## plotting results
    print(model.summary())
    loss_and_metrics(model, X_test, Y_test)
    plot_training_history(history)
    plot_predicted_vs_data(model, X_train, Y_train, label='In Sample', pred_window=prediction_window)

    plot_predicted_vs_data(model, X_test, Y_test, label='Out of Sample', pred_window=prediction_window)
    P.show()
