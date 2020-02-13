import pickle
import numpy as np
import scipy.stats as ss
from matplotlib import pyplot as P
import seaborn as sns


def predicted_vs_observed(predicted, real, city, state, doenca, plot=True):
    """
    Plot QQPlot for prediction values
    :param plot: generates an saves the qqplot when True (default)
    :param predicted: Predicted matrix
    :param real: Array of target_col values used in the prediction
    :param city: Geocode of the target city predicted
    :param state: State containing the city
    :param look_back: Look-back time window length used by the model
    :param all_predict_n: If True, plot the qqplot for every week predicted
    :return:
    """
    # Name = get_city_names([city])
    # data = get_alerta_table(city, state, doenca=doenca)

    obs_preds = np.hstack((predicted, real))
    q_p = [ss.percentileofscore(obs_preds, x) for x in predicted]
    q_o = [ss.percentileofscore(obs_preds, x) for x in real]
    plot_cross_qq(city, doenca, q_o, q_p)
    return np.array(q_o), np.array(q_p)


def plot_cross_qq(city, doenca, q_o, q_p):
    ax = sns.kdeplot(q_o[len(q_p) - len(q_o):], q_p, shade=True)
    ax.set_xlabel('observed')
    ax.set_ylabel('predicted')
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    P.plot([0, 100], [0, 100], 'k')
    P.title(f'Cross-prediction percentiles with {model} for {doenca} at {Name}.')
    P.savefig(f'cross_qqplot_{model}_{doenca}_{city}.png', dpi=300)


if __name__ == "__main__":
    model = 'lstm'
    # model = 'rqf'

    doença = 'chik'
    # city, Name = 3304557, 'Rio de Janeiro'
    # city, Name = 3303500, 'Nova Iguaçu'
    city, Name = 2304400, 'Fortaleza'
    if str(city).startswith('33'):
        state = 'RJ'
    elif str(city).startswith('23'):
        state = 'CE'

    # Weeks ahead to predict
    pw = 1

    if model == 'rqf':
        with open(f'../models/saved_models/quantile_forest/{state}/{city}_cross_{doença}_preditions.pkl', 'rb') as f:
            data = pickle.load(f)
            predicted = data['pred']
            real = data['target'][pw].values
    elif model == 'lstm':
        with open(f'../models/saved_models/LSTM/{state}/{city}_cross_{doença}_predictions.pkl', 'rb') as f:
            data = pickle.load(f)
            predicted = data['pred'][pw].values
            real = data['target'][:, pw]
    qp, qo = predicted_vs_observed(predicted, real, city, state, doença)
    print("Predicted: ", np.median(qp - qo), np.percentile(qp - qo, 25), np.percentile(qp - qo, 75))

    P.show()
