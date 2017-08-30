import numpy as np;
import pandas as pd
import scipy.spatial.distance as spd
from infodenguepredict.predict_settings import *
# from scipy.signal import correlate

from infodenguepredict.data.infodengue import get_alerta_table, combined_data

def get_cities_from_state(state):
    alerta_table = get_alerta_table(state=state)
    cities_list = alerta_table.municipio_geocodigo.unique()
    return cities_list


def alocate_data(state):
    cities_list = list(get_cities_from_state(state))
    bad_cities = []
    for city in cities_list:
        try:
            full_city = combined_data(city, data_types=data_types)
            full_city.to_pickle('{}/city_{}.pkl'.format(tmp_path, city))
        except TypeError as e:
            print("Skipping: ", city)
            bad_cities.append(city)
            continue
    for c in bad_cities:
        cities_list.remove(c)
    return cities_list


def correlation(df_1, df_2):
    corr_list = []
    for col in df_1.columns:
        df = pd.concat((df_1[col], df_2[col]), axis=1).fillna(method='ffill')
        corr = spd.pdist(df.T.as_matrix(), metric='correlation')
        corr_list.append(corr[0])
    return np.nanmean(corr_list)


def cross_correlation(df_1, df_2, max_lag=5):
    corr_list = []
    for col in df_1.columns:
        corrs = [np.correlate(df_1[col], df_2[col].shift(lag)) for lag in range(max_lag)]
        corr = np.argmax(corrs)
        lag = range(max_lag)[corrs.index(corr)]
        corr_list.append(corr)
    return np.nanmean(corr_list)


def distance(cities_list, cols):
    """
    returns the correlation distance matrix for a list of cities.
    :param cities_list: List of geocodes
    :param cols: columns to calculate the correlation
    :return:
    """
    state_distances = pd.DataFrame(index=cities_list)

    for pos, city_1 in enumerate(cities_list):
        print("Calculating distance Matrix for ", city_1)
        full_city_1 = pd.read_pickle('{}/city_{}.pkl'.format(tmp_path, city_1))[cols]
        new_col = list(np.zeros(pos + 1))

        for city_2 in cities_list[pos + 1:]:
            full_city_2 = pd.read_pickle('{}/city_{}.pkl'.format(tmp_path, city_2))[cols]

            dist = correlation(full_city_1, full_city_2)
            new_col.append(dist)
        state_distances[city_1] = new_col

    return state_distances

if __name__ == "__main__":
    cities_list = get_cities_from_state(state)
    distance(cities_list, cluster_vars)
