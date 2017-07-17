import numpy as np;
import pandas as pd
import scipy.spatial.distance as spd

from infodenguepredict.data.infodengue import get_alerta_table, combined_data


def get_cities_from_state(state):
    alerta_table = get_alerta_table(state=state)
    cities_list = alerta_table.municipio_geocodigo.unique()
    return cities_list


def alocate_data(state):
    cities_list = get_cities_from_state(state)
    for city in cities_list:
        full_city = combined_data(city).dropna()
        full_city.to_pickle('city_{}.pkl'.format(city))
    return cities_list


def correlation(df_1, df_2):
    corr_list = []
    for col in df_1.columns:
        df = pd.concat((df_1[col], df_2[col]), axis=1).fillna(method='ffill')
        corr = spd.pdist(df.T.as_matrix(), metric='correlation')
        corr_list.append(corr[0])
    return np.nanmean(corr_list)


def distance(cities_list):
    state_distances = pd.DataFrame(index=cities_list)

    cols = ['casos', 'p_rt1', 'p_inc100k', 'numero', 'temp_min',
            'temp_max', 'umid_min', 'pressao_min']

    for pos, city_1 in enumerate(cities_list):
        print(city_1)
        full_city_1 = pd.read_pickle('city_{}.pkl'.format(city_1))[cols]
        new_col = list(np.zeros(pos + 1))
        for city_2 in cities_list[pos + 1:]:
            full_city_2 = pd.read_pickle('city_{}.pkl'.format(city_2))[cols]

            dist = correlation(full_city_1, full_city_2)
            new_col.append(dist)
        state_distances[city_1] = new_col

    return state_distances



cities_list = alocate_data("RJ")
dists = distance(cities_list)
dists.to_pickle('dists_matrix.pkl')