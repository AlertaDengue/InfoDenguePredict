import numpy as np
import pandas as pd
import pickle
import math
import string as str
from matplotlib import pyplot as P
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard
from keras import backend as K
from hyperas.distributions import uniform, choice
from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe

from time import time
from infodenguepredict.data.infodengue import combined_data, get_cluster_data, random_data, get_city_names
from infodenguepredict.models.deeplearning.preprocessing import split_data, normalize_data
from infodenguepredict.predict_settings import *


def optimize_model(x_train, y_train, x_test, y_test, features):
    model = Sequential()

    model.add(LSTM({{choice([4, 8, 16])}}, input_shape=({{choice([2, 3, 4])}}, features), stateful=True,
                   batch_input_shape=(1, {{choice([2, 3, 4])}}, features),
                   return_sequences=True,
                   dropout={{uniform(0, 1)}},
                   recurrent_dropout={{uniform(0, 1)}}
                   ))
    model.add(LSTM({{choice([4, 8, 16])}}, input_shape=({{choice([2, 3, 4])}}, features), stateful=True,
                   batch_input_shape=(1, {{choice([2, 3, 4])}}, features),
                   return_sequences=True,
                   dropout={{uniform(0, 1)}},
                   recurrent_dropout={{uniform(0, 1)}}
                   ))

    model.add(LSTM({{choice([4, 8, 16])}}, input_shape=({{choice([2, 3, 4])}}, features), stateful=True,
                   batch_input_shape=(1, {{choice([2, 3, 4])}}, features),
                   dropout={{uniform(0, 1)}},
                   recurrent_dropout={{uniform(0, 1)}}
                   ))

    model.add(Dense(prediction_window, activation='relu'))

    start = time()
    model.compile(loss="msle", optimizer="rmsprop", )
    model.fit(x_train, y_train,
              batch_size=1,
              nb_epoch=1,
              validation_split=0.05,
              verbose=0,
              )
    loss = model.evaluate(x_test, y_test, batch_size=1)
    return {'loss': loss, 'status': STATUS_OK, 'model': model}


def build_model(hidden, features, predict_n, look_back=10, batch_size=1):
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
                   # activation='relu',
                   dropout=0.2,
                   recurrent_dropout=0.2,
                   implementation=2,
                   unit_forget_bias=True
                   ))
    model.add(LSTM(hidden, input_shape=(look_back, features), stateful=True,
                   batch_input_shape=(batch_size, look_back, features),
                   return_sequences=True,
                   # activation='relu',
                   dropout=0.2,
                   recurrent_dropout=0.2,
                   implementation=2,
                   unit_forget_bias=True
                   ))

    model.add(LSTM(hidden, input_shape=(look_back, features), stateful=True,
                   batch_input_shape=(batch_size, look_back, features),
                   # activation='relu',
                   dropout=0.2,
                   recurrent_dropout=0.2,
                   implementation=2,
                   unit_forget_bias=True
                   ))

    model.add(Dense(predict_n, activation='relu',
                    kernel_initializer='random_uniform',
                    bias_initializer='zeros'))

    start = time()
    model.compile(loss="msle", optimizer="nadam", metrics=['accuracy', 'mape'])
    print("Compilation Time : ", time() - start)
    # plot_model(model, to_file='LSTM_model.png')
    print(model.summary())
    return model


def train(model, X_train, Y_train, batch_size=1, epochs=10, geocode=None, overwrite=True):
    TB_callback = TensorBoard(log_dir='./tensorboard',
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True,
                              # embeddings_freq=10
                              )

    hist = model.fit(X_train, Y_train,
                     batch_size=batch_size,
                     nb_epoch=epochs,
                     validation_split=0.15,
                     verbose=1,
                     callbacks=[TB_callback])
    with open('history_{}.pkl'.format(geocode), 'wb') as f:
        pickle.dump(hist.history, f)
    model.save_weights('trained_{}_model.h5'.format(geocode), overwrite=overwrite)
    return hist


def plot_training_history(hist):
    """
    Plot the Loss series from training the model
    :param hist: Training history object returned by "model.fit()"
    """
    df_vloss = pd.DataFrame(hist.history['val_loss'], columns=['val_loss'])
    df_loss = pd.DataFrame(hist.history['loss'], columns=['loss'])
    df_mape = pd.DataFrame(hist.history['mean_absolute_percentage_error'], columns=['mape'])
    ax = df_vloss.plot(logy=True);
    df_loss.plot(ax=ax, grid=True, logy=True);
    # df_mape.plot(ax=ax, grid=True, logy=True);
    P.savefig("LSTM_training_history.png")


def plot_predicted_vs_data(predicted, Ydata, indice, label, pred_window, factor, split_point=None):
    """
    Plot the model's predictions against dat
    :param predicted:
    :param Ydata:
    :param indice:
    :param label:
    :param pred_window:
    :param factor:
    """

    P.clf()
    df_predicted = pd.DataFrame(predicted).T
    ymax = max(predicted.max() * factor, Ydata.max() * factor)
    P.vlines(indice[split_point], 0, ymax, 'g', 'dashdot', lw=2)
    P.text(indice[split_point + 2], 0.6*ymax, "Out of sample Predictions")
    for n in range(df_predicted.shape[1] - pred_window):
        P.plot(indice[n: n + pred_window], pd.DataFrame(Ydata.T)[n] * factor, 'k-')
        P.plot(indice[n: n + pred_window], df_predicted[n] * factor, 'r-.')
        P.vlines(indice[n: n + pred_window], np.zeros(pred_window), df_predicted[n] * factor, 'b', alpha=0.2)
    P.grid()
    P.title(label)
    P.xlabel('time')
    P.ylabel('incidence')
    P.xticks(rotation=70)
    P.legend(['data', 'predicted'])
    P.savefig("lstm_{}.png".format(label), bbox_inches='tight', dpi=300)
    P.show()


def loss_and_metrics(model, Xtest, Ytest):
    print(model.evaluate(Xtest, Ytest, batch_size=1))


def evaluate(city, model, Xdata, Ydata, label):
    loss_and_metrics(model, Xdata, Ydata)
    metrics = model.evaluate(Xdata, Ydata, batch_size=1)
    with open('metrics_{}.pkl'.format(label), 'wb') as f:
        pickle.dump(metrics, f)
    predicted = model.predict(Xdata, batch_size=1, verbose=1)
    return predicted, metrics


def train_evaluate_model(city, data, predict_n, look_back, hidden, epochs, cluster=True):
    if cluster:
        target_col = list(data.columns).index('casos_{}'.format(city))
    else:
        target_col = list(data.columns).index('casos')
    norm_data, max_features = normalize_data(data)

    ##split test and train
    X_train, Y_train, X_test, Y_test = split_data(norm_data,
                                                  look_back=look_back, ratio=.7,
                                                  predict_n=predict_n, Y_column=target_col)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


    ## Run model
    model = build_model(hidden, X_train.shape[2], predict_n=predict_n, look_back=look_back)
    history = train(model, X_train, Y_train, batch_size=1, epochs=epochs, geocode=city)
    # plot_training_history(history)
    # model.save('lstm_model')

    predicted_out, metrics_out = evaluate(city, model, X_test, Y_test, label='out_of_sample_{}'.format(city))
    predicted_in, metrics_in = evaluate(city, model, X_train, Y_train, label='in_sample_{}'.format(city))

    predicted = np.concatenate((predicted_in, predicted_out), axis=0)
    factor = max_features[target_col]

    return predicted, Y_test, Y_train, factor


def single_prediction(city, state, predictors, predict_n, look_back, hidden, epochs, random=False):
    """
    Fit an LSTM model to generate predictions for a city, Using its cluster as regressors.
    :param city: geocode of the target city
    :param state: State containing the city
    :param predict_n: How many weeks ahead to predict
    :param look_back: Look-back time window length used by the model
    :param hidden: Number of hidden layers in each LSTM unit
    :param epochs: Number of epochs of training
    :param random: If the model should be trained on a random selection of ten cities of the same state.
    :return:
    """
    if random == True:
        data, group = random_data(10, state, city)
    else:
        with open('../clusters_{}.pkl'.format(state), 'rb') as fp:
            clusters = pickle.load(fp)
        data, group = get_cluster_data(geocode=city, clusters=clusters,
                                       data_types=data_types, cols=predictors)

    indice = list(data.index)
    indice = [i.date() for i in indice]

    city_name = get_city_names([city, 0])[0][1]
    predicted, Y_test, Y_train, factor = train_evaluate_model(city, data, predict_n, look_back, hidden, epochs)

    plot_predicted_vs_data(predicted,
                           np.concatenate((Y_train, Y_test), axis=0),
                           indice[:],
                           label='Predictions for {}'.format(city_name),
                           pred_window=predict_n,
                           factor= factor,
                           split_point=len(Y_train))

    return None


def cluster_prediction(geocode, state, predictors, predict_n, look_back, hidden, epochs):
    """
    Fit an LSTM model to generate predictions for all cities from a cluster, Using its cluster as regressors.
    :param city: geocode of the target city
    :param state: State containing the city
    :param predict_n: How many weeks ahead to predict
    :param look_back: Look-back time window length used by the model
    :param hidden: Number of hidden layers in each LSTM unit
    :param epochs: Number of epochs of training
    :return:
    """

    clusters = pd.read_pickle('../../analysis/clusters_{}.pkl'.format(state))
    data, cluster = get_cluster_data(geocode=geocode, clusters=clusters,
                                       data_types=data_types, cols=predictors)
    indice = list(data.index)
    indice = [i.date() for i in indice]


    fig, axs = P.subplots(nrows=2, ncols=2, figsize=(50, 45))

    targets = zip(cluster, axs.flatten())
    for (city, ax) in targets:
        print(city)
        city_name = get_city_names([city, 0])[0][1]
        predicted, Y_test, Y_train, factor = train_evaluate_model(city, data, predict_n, look_back, hidden, epochs)

        ## plot
        Ydata = np.concatenate((Y_train, Y_test), axis=0)
        split_point = len(Y_train)
        df_predicted = pd.DataFrame(predicted).T
        ymax = max(predicted.max() * factor, Ydata.max() * factor)

        ax.vlines(indice[split_point], 0, ymax, 'g', 'dashdot', lw=2)
        ax.text(indice[split_point + 1], 0.6 * ymax, "Out of sample Predictions")
        for n in range(df_predicted.shape[1] - predict_n):
            ax.plot(indice[n: n + predict_n], pd.DataFrame(Ydata.T)[n] * factor, 'k-')
            ax.plot(indice[n: n + predict_n], df_predicted[n] * factor, 'r-')
            ax.vlines(indice[n: n + predict_n], np.zeros(predict_n), df_predicted[n] * factor, 'b', alpha=0.2)

        ax.grid()
        ax.set_title('Predictions for {}'.format(city_name), fontsize=13)
        ax.legend(['data', 'predicted'])

    P.tight_layout()
    P.savefig('cluster_{}.png'.format(geocode), dpi=300)#, bbox_inches='tight')
    P.show()

    return None


if __name__ == "__main__":
    # K.set_epsilon(1e-5)
    # single_prediction(city, state, predictors, predict_n=prediction_window, look_back=LOOK_BACK,
    #                   hidden=HIDDEN, epochs=epochs)

    cluster_prediction(city, state, predictors, predict_n=prediction_window, look_back=LOOK_BACK, hidden=HIDDEN, epochs=epochs)

    ## Optimize Hyperparameters
    #
    # def get_data():
    #     return X_train, Y_train, X_test, Y_test, X_train.shape[2]

    # best_run, best_model = optim.minimize(model=optimize_model,
    #                                       data=get_data,
    #                                       algo=tpe.suggest,
    #                                       max_evals=5,
    #                                       trials=Trials())
