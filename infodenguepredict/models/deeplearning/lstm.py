import numpy as np
import pandas as pd
import pickle
import math
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
from infodenguepredict.data.infodengue import get_alerta_table, get_temperature_data, get_tweet_data, get_cluster_data, random_data
from infodenguepredict.models.deeplearning.preprocessing import split_data, normalize_data



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
    model.compile(loss="mse", optimizer="rmsprop", )
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
                   dropout=0.2,
                   recurrent_dropout=0.2,
                   ))
    model.add(LSTM(hidden, input_shape=(look_back, features), stateful=True,
                   batch_input_shape=(batch_size, look_back, features),
                   return_sequences=True,
                   dropout=0.2,
                   recurrent_dropout=0.2,
                   ))

    model.add(LSTM(hidden, input_shape=(look_back, features), stateful=True,
                   batch_input_shape=(batch_size, look_back, features),
                   dropout=0.2,
                   recurrent_dropout=0.2
                   ))

    model.add(Dense(prediction_window, activation='relu'))

    start = time()
    model.compile(loss="poisson", optimizer="nadam", metrics=['accuracy', 'mape'])
    print("Compilation Time : ", time() - start)
    # plot_model(model, to_file='LSTM_model.png')
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
    with open('history_{}.pkl'.format(geocode),'wb') as f:
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



def plot_predicted_vs_data(model, Xdata, Ydata, label, pred_window, factor):
    P.clf()
    metrics = model.evaluate(Xdata, Ydata, batch_size=1)
    with open('metrics_{}.pkl'.format(label),'wb') as f:
        pickle.dump(metrics, f)
    predicted = model.predict(Xdata, batch_size=BATCH_SIZE, verbose=1)
    df_predicted = pd.DataFrame(predicted).T
    for n in range(df_predicted.shape[1]):
        P.plot(range(n, n + pred_window), pd.DataFrame(Ydata.T)[n]*factor, 'k-')
        P.plot(range(n, n + pred_window), df_predicted[n]*factor, 'g:o', alpha=0.5)
    P.grid()
    P. title(label)
    P.xlabel('weeks')
    P.ylabel('incidence')
    P.legend([label, 'predicted'])
    P.savefig("lstm_{}.png".format(label))


def loss_and_metrics(model, Xtest, Ytest):
    print(model.evaluate(Xtest, Ytest, batch_size=1))


def single_prediction(city, state, predict_n, time_window, hidden, random=False):
    codes = pd.read_excel('../../data/codigos_rj.xlsx', names=['city', 'code'], header=None).set_index('code').T

    if random==True:
        data, group = random_data(10, state, city)
    else:
        with open('../clusters_{}.pkl'.format(state), 'rb') as fp:
            clusters = pickle.load(fp)
        # data = get_example_table(3304557) #Nova Iguaçu: 3303500
        # data = get_complete_table(3304557)
        # data = build_multicity_dataset('RJ')
        data, group = get_cluster_data(city, clusters)

    cluster = [codes[i] for i in group]
    target_col = list(data.columns).index('casos_{}'.format(city))
    norm_data, max_features = normalize_data(data)
    # P.show()
    X_train, Y_train, X_test, Y_test = split_data(norm_data,
                                                  look_back=time_window, ratio=.7,
                                                  predict_n=predict_n, Y_column=target_col)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    ## Run model
    model = build_model(hidden, X_train.shape[2], time_window, 1)
    history = train(model, X_train, Y_train, batch_size=1, epochs=epochs, geocode=city)
    # model.save('lstm_model')

    ## plotting results
    loss_and_metrics(model, X_test, Y_test)
    plot_training_history(history)
    plot_predicted_vs_data(model, X_train, Y_train, label='In Sample {}'.format(city), pred_window=predict_n,
                           factor=max_features[target_col])
    plot_predicted_vs_data(model, X_test, Y_test, label='Out of Sample {}'.format(city), pred_window=predict_n,
                           factor=max_features[target_col])
    print(cluster)
    return None


def cluster_prediction(state, predict_n, time_window, hidden, epochs):
    codes = pd.read_excel('../../data/codigos_rj.xlsx', names=['city', 'code'], header=None).set_index('code').T

    with open('../clusters_{}.pkl'.format(state), 'rb') as fp:
        clusters = pickle.load(fp)

    for i, cluster in enumerate(clusters[3:]):
        # if i < 3: continue
        data, cluster_n = get_cluster_data(cluster[0], clusters)

        if len(cluster) < 9:
            fig, axs = P.subplots(nrows=math.ceil(len(cluster) / 3), ncols=3, figsize=(45, 45))
        else:
            fig, axs = P.subplots(nrows=math.ceil(len(cluster) / 4), ncols=4, figsize=(45, 45))

        targets = zip(cluster, axs.flatten())
        for (city, ax) in targets:
            target_col = list(data.columns).index('casos_{}'.format(city))
            norm_data, max_features = normalize_data(data)
            #model
            X_train, Y_train, X_test, Y_test = split_data(norm_data,
                                                          look_back=time_window, ratio=.7,
                                                          predict_n=predict_n, Y_column=target_col)
            model = build_model(hidden, X_train.shape[2], time_window, 1)
            history = train(model, X_train, Y_train, batch_size=1, epochs=epochs, geocode=city)

            #plot
            predicted = model.predict(X_test, batch_size=1, verbose=1)
            df_predicted = pd.DataFrame(predicted).T
            factor = max_features[target_col]
            for n in range(df_predicted.shape[1]):
                ax.plot(range(n, n + predict_n), pd.DataFrame(Y_test.T)[n]*factor, 'k-')
                ax.plot(range(n, n + predict_n), df_predicted[n]*factor, 'g:o', alpha=0.5)
            ax.grid()
            ax.set_title(codes[city])
        P.savefig('cluster_{}.png'.format(i), dpi=400)
        # P.show()

    return None



if __name__ == "__main__":
    TIME_WINDOW = 4
    HIDDEN = 64
    LOOK_BACK = 4
    BATCH_SIZE = 1
    prediction_window = 3  # weeks
    city = 3303500
    state = 'RJ'
    epochs = 1

    single_prediction(city, state, predict_n=prediction_window, time_window=TIME_WINDOW, hidden=HIDDEN, random=True)
    # cluster_prediction(state, predict_n=prediction_window, time_window=TIME_WINDOW, hidden=HIDDEN, epochs=epochs)

    ## Optimize Hyperparameters
    #
    # def get_data():
    #     return X_train, Y_train, X_test, Y_test, X_train.shape[2]

    # best_run, best_model = optim.minimize(model=optimize_model,
    #                                       data=get_data,
    #                                       algo=tpe.suggest,
    #                                       max_evals=5,
    #                                       trials=Trials())




