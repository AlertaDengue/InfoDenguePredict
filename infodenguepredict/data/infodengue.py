"""
This module performs data fetching from the Infodengue database.
for remote database access, we recommend establishing an SSH tunnel:
 ssh -f user@remote-server -L 5432:localhost:5432 -N
"""

import pandas as pd
import random
from sqlalchemy import create_engine
from decouple import config


db_engine = create_engine("postgresql://{}:{}@{}/{}".format(
    config('PSQL_USER'),
    config('PSQL_PASSWORD'),
    config('PSQL_HOST'),
    config('PSQL_DB')
))

def get_alerta_table(municipio=None, state=None):
    """
    Pulls the data from a single city, cities from a state or all cities from the InfoDengue
    database
    :param municipio: geocode (one city) or None (all)
    :return: Pandas dataframe
    """
    conexao = create_engine("postgresql://{}:{}@{}/{}".format(config('PSQL_USER'),
                                                              config('PSQL_PASSWORD'),
                                                              config('PSQL_HOST'),
                                                              config('PSQL_DB')))
    if municipio is None:
        if state == 'RJ':
            sql = 'select * from "Municipio"."Historico_alerta"  where municipio_geocodigo>3300000 and municipio_geocodigo<4000000 ORDER BY "data_iniSE", municipio_geocodigo ASC;'
        elif state == 'ES':
            sql = 'select * from "Municipio"."Historico_alerta"  where municipio_geocodigo>3200000 and municipio_geocodigo<3300000 ORDER BY "data_iniSE", municipio_geocodigo ASC;'
        elif state == 'PR':
            sql = 'select * from "Municipio"."Historico_alerta"  where municipio_geocodigo>4000000 and municipio_geocodigo<5000000 ORDER BY "data_iniSE", municipio_geocodigo ASC;'
        elif state is None:
            sql = 'select * from "Municipio"."Historico_alerta" ORDER BY "data_iniSE", municipio_geocodigo ASC;'
        else:
            raise NameError("{} is not a valid state identifier".format(state))
        df = pd.read_sql_query(sql, conexao, index_col='id')
    else:
        df = pd.read_sql_query(
            'select * from "Municipio"."Historico_alerta" where municipio_geocodigo={} ORDER BY "data_iniSE" ASC;'.format(
                municipio),
            conexao, index_col='id')
    df.data_iniSE = pd.to_datetime(df.data_iniSE)
    df.set_index('data_iniSE', inplace=True)
    return df


def get_temperature_data(municipio=None):
    """
    Fecth dataframe with temperature time series for a given city
    :param municipio: geocode of the city
    :return: pandas dataframe
    """
    conexao = create_engine("postgresql://{}:{}@{}/{}".format(config('PSQL_USER'),
                                                              config('PSQL_PASSWORD'),
                                                              config('PSQL_HOST'),
                                                              config('PSQL_DB')))

    if municipio is None:
        df = pd.read_sql_query('select * from "Municipio"."Clima_wu" ORDER BY "data_dia" ASC;',
                               conexao, index_col='id')
    else:
        df = pd.read_sql_query(
            'select cwu.id, temp_min, temp_max, umid_min, pressao_min, data_dia FROM "Municipio"."Clima_wu" cwu JOIN "Dengue_global".regional_saude rs ON cwu."Estacao_wu_estacao_id"=rs.codigo_estacao_wu WHERE rs.municipio_geocodigo={} ORDER BY data_dia ASC;'.format(
                municipio), conexao, index_col='id')
    df.data_dia = pd.to_datetime(df.data_dia)
    df.set_index('data_dia', inplace=True)
    return df


def get_tweet_data(municipio=None) -> pd.DataFrame:
    """
    Fetch Dataframe with dengue tweet time series for a given city
    :param municipio: city geocode.
    :return: pandas dataframe
    """
    conexao = create_engine("postgresql://{}:{}@{}/{}".format(config('PSQL_USER'),
                                                              config('PSQL_PASSWORD'),
                                                              config('PSQL_HOST'),
                                                              config('PSQL_DB')))
    if municipio is None:
        df = pd.read_sql_query('select * from "Municipio"."Tweet" ORDER BY "data_dia" ASC;',
                               conexao, index_col='id')
    else:
        df = pd.read_sql_query(
            'select * FROM "Municipio"."Tweet" WHERE "Municipio_geocodigo"={} ORDER BY data_dia ASC;'.format(municipio),
            conexao, index_col='id')
        del df['Municipio_geocodigo']
    df.data_dia = pd.to_datetime(df.data_dia)
    df.set_index('data_dia', inplace=True)
    return df


def get_rain_data(geocode, sensor="chuva"):
    """
    Return the series of rain data.  for all stations contained in the geocode
    :param geocode: geocode of a city
    :param sensor: either "chuva" or "intensidade_precipitaçao"
    :return: pandas dataframe.
    """
    conexao = create_engine("postgresql://{}:{}@{}/{}".format(config('PSQL_USER'),
                                                              config('PSQL_PASSWORD'),
                                                              config('PSQL_HOST'),
                                                              config('PSQL_DB')))

    sql = "SELECT * FROM \"Municipio\".\"Clima_cemaden\" WHERE \"Estacao_cemaden_codestacao\" similar to '{}%' and sensor='{}'".format(
        geocode, sensor)
    print(sql)
    df = pd.read_sql_query(sql, conexao, index_col='id')
    df.datahora = pd.to_datetime(df.datahora)
    df.set_index('datahora', inplace=True)
    return df

def get_city_names(geocodigos):
    """
    Fetch names of the cities from a list of geocodes.
    :param geocodigos: list of 7-digit geocodes.
    :return:
    """
    with db_engine.connect() as conexao:
        res = conexao.execute('select geocodigo, nome from "Dengue_global"."Municipio" WHERE geocodigo in {};'.format(tuple(geocodigos)))
        res = res.fetchall()

    return res


def build_multicity_dataset(state) -> pd.DataFrame:
    """
    Fetches a data table for the specfied state, and converts it from long to wide format,
    so that it can be fed straight to th models.
    :param state: Two letter code for the state
    :return: Panda DataFrame
    """
    full_data = get_alerta_table(state=state)
    for col in ['casos_est_min', 'casos_est_max', 'Localidade_id', 'versao_modelo', 'municipio_nome']:
        del full_data[col]
    full_data = full_data.pivot(index=full_data.index, columns='municipio_geocodigo')
    full_data.columns = ['{}_{}'.format(*col).strip() for col in full_data.columns.values]

    return full_data


def combined_data(municipio):
    """
    Returns combined dataframe with incidence, tweets, and temperature
    :param municipio: geocode
    :return: Dataframe
    """
    alerta_table = get_alerta_table(municipio=municipio)
    tweets = get_tweet_data(municipio)
    tweets = tweets.resample('W').sum()

    weather = get_temperature_data(municipio)
    weather = weather.resample('W').mean()

    full_data = pd.concat([alerta_table, tweets, weather], axis=1, join='inner')
    return full_data


def get_cluster_data(geocode, clusters):
    """

    :param geocode:
    :param clusters:
    :return:
    """
    to_drop = ['casos_est_min', 'casos_est_max', 'Localidade_id', 'versao_modelo',
               'municipio_nome', 'casos_est', 'municipio_geocodigo', 'nivel']

    cluster = list(filter(lambda x: geocode in x, clusters))[0]

    full_data = pd.DataFrame()
    for city_code in cluster:
        tmp = combined_data(city_code).drop(to_drop, axis=1)
        tmp.columns = ['{}_{}'.format(col, city_code) for col in tmp.columns.values]
        full_data = pd.concat([tmp, full_data], axis=1).fillna(method='ffill')

    return full_data, cluster


def get_example_table(geocode=None):
    """
    Fetch the data from the database, filters out useless variables
    :return: pandas dataframe
    """
    raw_df = get_alerta_table(geocode)
    filtered_df = raw_df[['SE', 'casos_est', 'casos_est_min', 'casos_est_max',
                          'casos', 'municipio_geocodigo', 'p_rt1', 'p_inc100k', 'nivel']]
    filtered_df['SE'] = [int(str(x)[-2:]) for x in filtered_df.SE]

    return filtered_df


# def get_complete_table(geocode=None):
#     """
#     Extends Example table with temperature, humidity atmospheric pressure and Tweets
#     :param geocode:
#     :return:
#     """
#     df = get_example_table(geocode=geocode)
#     T = get_temperature_data(geocode)
#     Tw = get_tweet_data(municipio=geocode)
#     Tw.pop('Municipio_geocodigo')
#     Tw.pop('CID10_codigo')
#     complete = df.join(T).join(Tw).dropna()
#     return complete


def random_data(N, state, city=None):
    """

    :param N:
    :param state:
    :param city:
    :return:
    """
    to_drop = ['casos_est_min', 'casos_est_max', 'Localidade_id', 'versao_modelo',
               'municipio_nome', 'casos_est', 'municipio_geocodigo', 'nivel']

    alerta_table = get_alerta_table(state=state)
    cities_list = alerta_table.municipio_geocodigo.unique()

    random_group = random.sample(list(cities_list), N)

    if city != None:
        if city not in random_group:
            random_group.append(city)

    full_data = pd.DataFrame()
    for city_code in random_group:
        tmp = combined_data(city_code).drop(to_drop, axis=1)
        tmp.columns = ['{}_{}'.format(col, city_code) for col in tmp.columns.values]
        full_data = pd.concat([tmp, full_data], axis=1).fillna(method='ffill')

    return full_data, random_group
