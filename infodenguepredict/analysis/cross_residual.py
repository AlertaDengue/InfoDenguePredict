import pickle
import numpy as np
import pandas as pd
import scipy.stats as ss
from matplotlib import pyplot as P
import seaborn as sns
from infodenguepredict.data.infodengue import get_alerta_table, get_city_names


def qqplot(predicted, real, city, state, doenca):
    """
    Plot QQPlot for prediction values
    :param predicted: Predicted matrix
    :param real: Array of target_col values used in the prediction
    :param city: Geocode of the target city predicted
    :param state: State containing the city
    :param look_back: Look-back time window length used by the model
    :param all_predict_n: If True, plot the qqplot for every week predicted
    :return:
    """
    # Name = get_city_names([city])
    data = get_alerta_table(city, state, doenca=doenca)

    target = real[1]
    obs_preds = np.hstack((predicted, target.values))
    q_p = [ss.percentileofscore(obs_preds, x) for x in predicted]
    q_o = [ss.percentileofscore(obs_preds, x) for x in target.values]
    ax = sns.kdeplot(q_o[len(q_p)-len(q_o):], q_p, shade=True)
    ax.set_xlabel('observed')
    ax.set_ylabel('predicted')
    ax.set_xlim([0,100])
    ax.set_ylim([0,100])

    P.plot([0, 100], [0, 100], 'k')

    P.title(f'Cross-prediction errors for {doenca} at {Name}.')
    P.savefig(f'cross_qqplot_{doenca}_{city}.png', dpi=300)
    # print(data.head())


if __name__ == "__main__":
    # state = 'RJ'
    state = 'CE'
    doença = 'chik'
    # city = 3304557
    # city = 3303500
    city = 2304400
    # Name = 'Rio de Janeiro'
    # Name = 'Nova Iguaçu'
    Name = 'Fortaleza'
    with open(f'../models/saved_models/quantile_forest/{state}/{city}_cross_{doença}_preditions.pkl', 'rb') as f:
        data = pickle.load(f)
    qqplot(data['pred'], data['target'], city, state, doença)
    P.show()
