"""
This module performs data fetching from the Infodengue database.
for remote database access, we recommend establishing an SSH tunnel:
 ssh -f user@remote-server -L 5432:localhost:5432 -N
"""

import pandas as pd
from sqlalchemy import create_engine
from decouple import config


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
        df = pd.read_sql_query('select * from "Municipio"."Historico_alerta" where municipio_geocodigo={} ORDER BY "data_iniSE" ASC;'.format(municipio),
                               conexao, index_col='id')
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
        df = pd.read_sql_query('select cwu.id, temp_min, umid_min, pressao_min, data_dia FROM "Municipio"."Clima_wu" cwu JOIN "Dengue_global".regional_saude rs ON cwu."Estacao_wu_estacao_id"=rs.codigo_estacao_wu WHERE rs.municipio_geocodigo={} ORDER BY data_dia ASC;'.format(municipio), conexao, index_col='id')
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
        df = pd.read_sql_query('select * FROM "Municipio"."Tweet" WHERE "Municipio_geocodigo"={} ORDER BY data_dia ASC;'.format(municipio), conexao, index_col='id')
    df.set_index('data_dia', inplace=True)
    return df

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
