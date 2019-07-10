import numpy as np
import pandas as pd
import pickle
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Tesla K40

from matplotlib import pyplot as P
import keras
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard, EarlyStopping
from keras import backend as K
from sklearn.metrics import *

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


def build_model(hidden, features, predict_n, look_back=10, batch_size=1):
    """
    Builds and returns the LSTM model with the parameters given
    :param hidden: number of hidden nodes
    :param features: number of variables in the example table
    :param look_back: Number of time-steps to look back before predicting
    :param batch_size: batch size for batch training
    :return:
    """
    inp = keras.Input(shape=(look_back, features), batch_shape=(batch_size, look_back, features))
    x = LSTM(
            hidden,
            input_shape=(look_back, features),
            stateful=True,
            batch_input_shape=(batch_size, look_back, features),
            return_sequences=True,
            # activation='relu',
            dropout=0,
            recurrent_dropout=0,
            implementation=2,
            unit_forget_bias=True,
        )(inp, training=True)
    x = Dropout(0.2)(x, training=True)
    x = LSTM(
            hidden,
            input_shape=(look_back, features),
            stateful=True,
            batch_input_shape=(batch_size, look_back, features),
            return_sequences=True,
            # activation='relu',
            dropout=0,
            recurrent_dropout=0,
            implementation=2,
            unit_forget_bias=True,
        )(x, training=True)
    x = Dropout(0.2)(x, training=True)
    x = LSTM(
            hidden,
            input_shape=(look_back, features),
            stateful=True,
            batch_input_shape=(batch_size, look_back, features),
            # activation='relu',
            dropout=0,
            recurrent_dropout=0,
            implementation=2,
            unit_forget_bias=True,
        )(x, training=True)
    x = Dropout(0.2)(x, training=True)
    out = Dense(
            predict_n,
            activation="relu",
            kernel_initializer="random_uniform",
            bias_initializer="zeros",
        )(x)
    model = keras.Model(inp, out)


    # model = Sequential()
    #
    # model.add(
    #     LSTM(
    #         hidden,
    #         input_shape=(look_back, features),
    #         stateful=True,
    #         batch_input_shape=(batch_size, look_back, features),
    #         return_sequences=True,
    #         # activation='relu',
    #         dropout=0,
    #         recurrent_dropout=0,
    #         implementation=2,
    #         unit_forget_bias=True,
    #     )
    # )
    # model.add(Dropout(0.2))
    # model.add(
    #     LSTM(
    #         hidden,
    #         input_shape=(look_back, features),
    #         stateful=True,
    #         batch_input_shape=(batch_size, look_back, features),
    #         return_sequences=True,
    #         # activation='relu',
    #         dropout=0,
    #         recurrent_dropout=0,
    #         implementation=2,
    #         unit_forget_bias=True,
    #     )
    # )
    # model.add(Dropout(0.2))
    # model.add(
    #     LSTM(
    #         hidden,
    #         input_shape=(look_back, features),
    #         stateful=True,
    #         batch_input_shape=(batch_size, look_back, features),
    #         # activation='relu',
    #         dropout=0,
    #         recurrent_dropout=0,
    #         implementation=2,
    #         unit_forget_bias=True,
    #     )
    # )
    # model.add(Dropout(0.2))
    # model.add(
    #     Dense(
    #         predict_n,
    #         activation="relu",
    #         kernel_initializer="random_uniform",
    #         bias_initializer="zeros",
    #     )
    # )

    start = time()
    model.compile(loss="msle", optimizer="nadam", metrics=["accuracy", "mape", "mse"])
    print("Compilation Time : ", time() - start)
    plot_model(model, to_file="LSTM_model.png")
    print(model.summary())
    return model


def train(
        model, X_train, Y_train, batch_size=1, epochs=10, geocode=None, overwrite=True
):
    TB_callback = TensorBoard(
        log_dir="./tensorboard",
        histogram_freq=0,
        write_graph=True,
        write_images=True,
        # embeddings_freq=10
    )

    hist = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        nb_epoch=epochs,
        validation_split=0.15,
        verbose=1,
        callbacks=[TB_callback, EarlyStopping(patience=10)]
    )
    with open("history_{}.pkl".format(geocode), "wb") as f:
        pickle.dump(hist.history, f)
    model.save_weights("trained_{}_model.h5".format(geocode), overwrite=overwrite)
    return hist


def plot_training_history(hist):
    """
    Plot the Loss series from training the model
    :param hist: Training history object returned by "model.fit()"
    """
    df_vloss = pd.DataFrame(hist.history["val_loss"], columns=["val_loss"])
    df_loss = pd.DataFrame(hist.history["loss"], columns=["loss"])
    df_mape = pd.DataFrame(
        hist.history["mean_absolute_percentage_error"], columns=["mape"]
    )
    ax = df_vloss.plot(logy=True)
    df_loss.plot(ax=ax, grid=True, logy=True)
    # df_mape.plot(ax=ax, grid=True, logy=True);
    # P.savefig("{}/LSTM_training_history.png".format(FIG_PATH))


def plot_predicted_vs_data(predicted, Ydata, indice, label, pred_window, factor, split_point=None):
    """
    Plot the model's predictions against data
    :param predicted: model predictions
    :param Ydata: observed data
    :param indice:
    :param label: Name of the locality of the predictions
    :param pred_window:
    :param factor: Normalizing factor for the target variable
    """

    P.clf()
    df_predicted = pd.DataFrame(predicted).T
    ymax = max(predicted.max() * factor, Ydata.max() * factor)
    P.vlines(indice[split_point], 0, ymax, "g", "dashdot", lw=2)
    P.text(indice[split_point + 2], 0.6 * ymax, "Out of sample Predictions")

    # plot only the last prediction point
    x = []
    y = []
    for n in range(df_predicted.shape[1] - pred_window):
        P.plot(
            indice[n: n + pred_window],
            pd.DataFrame(Ydata.T)[n] * factor,
            "k-",
            alpha=0.7,
        )
        P.vlines(
            indice[n + pred_window], 0, df_predicted[n][3] * factor, "b", alpha=0.2
        )
        x.append(indice[n + pred_window])
        y.append(df_predicted[n][3] * factor)
    P.plot(x, y, "r-", alpha=0.7)

    # plot all predicted points
    # P.plot(indice[pred_window:], pd.DataFrame(Ydata)[7] * factor, 'k-')
    # for n in range(df_predicted.shape[1] - pred_window):
    #     P.plot(
    #         indice[n: n + pred_window],
    #         pd.DataFrame(Ydata.T)[n] * factor,
    #         "k-",
    #         alpha=0.7,
    #     )
    #     P.plot(indice[n: n + pred_window], df_predicted[n] * factor, "r-")
    #     try:
    #         P.vlines(
    #             indice[n + pred_window],
    #             0,
    #             df_predicted[n].values[-1] * factor,
    #             "b",
    #             alpha=0.2,
    #         )
    #     except IndexError as e:
    #         print(indice.shape, n, df_predicted.shape)

    P.grid()
    P.title("Predictions for {}".format(label))
    P.xlabel("time")
    P.ylabel("incidence")
    P.xticks(rotation=70)
    P.legend(["data", "predicted"])
    P.savefig(
        "../saved_models/LSTM/{}/lstm_{}_ss.png".format(STATE, label),
        bbox_inches="tight",
        dpi=300,
    )
    P.show()


def loss_and_metrics(model, Xtest, Ytest):
    print(model.evaluate(Xtest, Ytest, batch_size=1))


def evaluate(city, model, Xdata, Ydata, label, uncertainty=True):
    loss_and_metrics(model, Xdata, Ydata)
    metrics = model.evaluate(Xdata, Ydata, batch_size=1)
    # with open('metrics_{}.pkl'.format(label), 'wb') as f:
    #     pickle.dump(metrics, f)
    if uncertainty:
        predicted  = []
    else:
        predicted = model.predict(Xdata, batch_size=1, verbose=1)
    return predicted, metrics


def calculate_metrics(pred, ytrue, factor):
    metrics = pd.DataFrame(
        index=(
            "mean_absolute_error",
            "explained_variance_score",
            "mean_squared_error",
            "mean_squared_log_error",
            "median_absolute_error",
            "r2_score",
        )
    )
    for col in range(pred.shape[1]):
        y = ytrue[:, col] * factor
        p = pred[:, col] * factor
        l = [
            mean_absolute_error(y, p),
            explained_variance_score(y, p),
            mean_squared_error(y, p),
            mean_squared_log_error(y, p),
            median_absolute_error(y, p),
            r2_score(y, p),
        ]
        metrics[col] = l
    return metrics


def train_evaluate_model(city, data, predict_n, look_back, hidden, epochs, ratio=0.7, cluster=True, load=False):
    """
    Train the model
    :param city: Name of the city
    :param data: Dataset
    :param predict_n: Number of steps ahead to be predicted
    :param look_back: number of history steps to include in training window
    :param hidden: Number of Hidden layer
    :param epochs: number of training epochs
    :param ratio: ratio of the full dataset to use in training
    :param cluster: whether to train on features from the city's cluster
    :param load: Whether to load a previously saved model
    :return:
    """
    if cluster:
        target_col = list(data.columns).index("casos_est_{}".format(city))
    else:
        target_col = list(data.columns).index("casos_est")
    norm_data, max_features = normalize_data(data)
    factor = max_features[target_col]

    ##split test and train
    X_train, Y_train, X_test, Y_test = split_data(
        norm_data,
        look_back=look_back,
        ratio=ratio,
        predict_n=predict_n,
        Y_column=target_col,
    )
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    ## Run model
    model = build_model(
        hidden, X_train.shape[2], predict_n=predict_n, look_back=look_back
    )
    if load:
        model.load_weights("trained_{}_model.h5".format(city))
    history = train(model, X_train, Y_train, batch_size=1, epochs=epochs, geocode=city)
    # model.save('../saved_models/lstm_{}_epochs_{}.h5'.format(city, epochs))

    predicted_out, metrics_out = evaluate(
        city, model, X_test, Y_test, label="out_of_sample_{}".format(city)
    )
    predicted_in, metrics_in = evaluate(
        city, model, X_train, Y_train, label="in_sample_{}".format(city)
    )

    metrics = calculate_metrics(predicted_out, Y_test, factor)
    metrics.to_pickle(
        "../saved_models/LSTM/{}/metrics_lstm_{}_8pw.pkl".format(STATE, city)
    )

    predicted = np.concatenate((predicted_in, predicted_out), axis=0)
    with open(
            "../saved_models/LSTM/{}/predicted_lstm_{}_8pw.pkl".format(STATE, city), "wb"
    ) as f:
        pickle.dump(predicted, f)

    return predicted, X_test, Y_test, Y_train, factor


def single_prediction(city, state, predictors, predict_n, look_back, hidden, epochs, predict=False):
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

    with open("../../analysis/clusters_{}.pkl".format(state), "rb") as fp:
        clusters = pickle.load(fp)
    data, group = get_cluster_data(
        geocode=city, clusters=clusters, data_types=DATA_TYPES, cols=predictors
    )

    indice = list(data.index)
    indice = [i.date() for i in indice]

    city_name = get_city_names([city, 0])[0][1]
    if predict:
        ratio = 1
    else:
        ratio = 0.7

    predicted, X_test, Y_test, Y_train, factor = train_evaluate_model(
        city, data, predict_n, look_back, hidden, epochs, ratio=ratio, load=False
    )
    plot_predicted_vs_data(
        predicted,
        np.concatenate((Y_train, Y_test), axis=0),
        indice[:],
        label="{}".format(city_name),
        pred_window=predict_n,
        factor=factor,
        split_point=len(Y_train),
    )

    return predicted, X_test, Y_test, Y_train, factor


def cluster_prediction(
        geocode, state, predictors, predict_n, look_back, hidden, epochs
):
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

    clusters = pd.read_pickle("../../analysis/clusters_{}.pkl".format(state))
    if os.path.exists('{}_cluster.csv'.format(geocode)):
        data = pd.read_csv('{}_cluster.csv.gz')
        cluster = pickle.load('{}_cluster.pkl')
    else:
        data, cluster = get_cluster_data(
            geocode=geocode, clusters=clusters, data_types=DATA_TYPES, cols=predictors, save=True
        )
    indice = list(data.index)
    indice = [i.date() for i in indice]

    fig, axs = P.subplots(nrows=2, ncols=2, figsize=(50, 45))

    targets = zip(cluster, axs.flatten())
    for (city, ax) in targets:
        print(city)
        city_name = get_city_names([city, 0])[0][1]
        predicted, X_test, Y_test, Y_train, factor = train_evaluate_model(
            city, data, predict_n, look_back, hidden, epochs
        )

        ## plot
        Ydata = np.concatenate((Y_train, Y_test), axis=0)
        split_point = len(Y_train)
        df_predicted = pd.DataFrame(predicted).T
        ymax = max(predicted.max() * factor, Ydata.max() * factor)

        ax.vlines(indice[split_point], 0, ymax, "g", "dashdot", lw=2)
        ax.text(indice[split_point + 1], 0.6 * ymax, "Out of sample Predictions")
        for n in range(df_predicted.shape[1] - predict_n):
            ax.plot(indice[n: n + predict_n], pd.DataFrame(Ydata.T)[n] * factor, "k-")
            ax.plot(indice[n: n + predict_n], df_predicted[n] * factor, "r-")
            ax.vlines(
                indice[n: n + predict_n],
                np.zeros(predict_n),
                df_predicted[n] * factor,
                "b",
                alpha=0.2,
            )

        ax.grid()
        ax.set_title("Predictions for {}".format(city_name), fontsize=13)
        ax.legend(["data", "predicted"])

    P.tight_layout()
    P.savefig("{}/cluster_{}.pdf".format(FIG_PATH, geocode))  # , bbox_inches='tight')
    # P.show()

    return None


def state_prediction(
        state, predictors, predict_n, look_back, hidden, epochs, predict=False
):
    clusters = pd.read_pickle("../../analysis/clusters_{}.pkl".format(state))

    for cluster in clusters:
        data, group = get_cluster_data(
            geocode=cluster[0],
            clusters=clusters,
            data_types=DATA_TYPES,
            cols=predictors,
        )
        for city in cluster:
            if os.path.exists(
                    "../saved_models/LSTM/{}/predicted_lstm_{}.pkl".format(state, city)
            ):
                continue

            indice = list(data.index)
            indice = [i.date() for i in indice]

            city_name = get_city_names([city, 0])[0][1]
            if predict:
                ratio = 1
            else:
                ratio = 0.7

            predicted, X_test, Y_test, Y_train, factor = train_evaluate_model(
                city, data, predict_n, look_back, hidden, epochs, ratio=ratio
            )
            plot_predicted_vs_data(
                predicted,
                np.concatenate((Y_train, Y_test), axis=0),
                indice[:],
                label=city_name,
                pred_window=predict_n,
                factor=factor,
                split_point=len(Y_train),
            )
            print("{} done".format(city))
    return None


if __name__ == "__main__":
    # K.set_epsilon(1e-5)

    single_prediction(
        CITY,
        STATE,
        PREDICTORS,
        predict_n=PREDICTION_WINDOW,
        look_back=LOOK_BACK,
        hidden=HIDDEN,
        epochs=EPOCHS,
    )

    # cluster_prediction(CITY, STATE, PREDICTORS, predict_n=PREDICTION_WINDOW, look_back=LOOK_BACK, hidden=HIDDEN,
    #                    epochs=EPOCHS)
