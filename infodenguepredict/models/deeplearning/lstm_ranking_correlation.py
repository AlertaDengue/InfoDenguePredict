from infodenguepredict.data.infodengue import build_multicity_dataset, get_cluster_data
from infodenguepredict.models.deeplearning.lstm import single_prediction
import numpy as np
import pandas as pd
import re
import pickle

def rank_cities(state):
    mult = build_multicity_dataset(state)
    cols = list(filter(re.compile('casos_\d+').search, mult.columns))
    mult = mult[cols]

    print(mult.head())
    codes = pd.read_excel('../../data/codigos_{}.xlsx'.format(state),
                          names=['city', 'code'], header=None).set_index('code').T

    ints = pd.DataFrame()
    for col in mult.columns:
        # ints.loc[codes[int(re.sub('casos_', '', col))]] = [np.trapz(mult[col])]
        ints[col] = [np.trapz(mult[col])]
    return ints


if __name__ == "__main__":
    TIME_WINDOW = 4
    HIDDEN = 4
    LOOK_BACK = 4
    BATCH_SIZE = 1
    prediction_window = 3  # weeks
    # city = 3303500
    state = 'RJ'
    epochs = 10

    rank = rank_cities(state)

    mapes = []

    for col in rank:
        city = re.sub('casos_', '', col)
        metric = single_prediction(int(city), state, predict_n=prediction_window, time_window=TIME_WINDOW, hidden=HIDDEN,
                                   epochs=epochs)
        mapes.append(metric)

    rank = rank.T
    rank['mape'] = mapes
    rank.to_pickle('rank.pkl')