"""
This module performs data fetching from the Infodengue database.
for remote database access, we recommend establishing an SSH tunnel:
ssh -f user@remote-server -L 5432:localhost:5432 -N
"""

import pandas as pd
from sqlalchemy import create_engine
from decouple import config

def get_alerta_table(municipio=None):
    """
    pulls the data from a single city or all cities from the InfoDengue
    database
    :param municipio: geocode (one city) or None (all)
    :return: Pandas dataframe
    """
    conexao = create_engine("postgresql://{}:{}@{}/{}".format(config('PSQL_USER'),
                                          config('PSQL_PASSWORD'),
                                          config('PSQL_HOST'),
                                          config('PSQL_DB')))
    if municipio is None:
        df = pd.read_sql_query('select * from "Municipio"."Historico_alerta" ORDER BY "data_iniSE" ASC;',
                                conexao, index_col='id')
    else:
        df = pd.read_sql_query('select * from "Municipio"."Historico_alerta" where municipio_geocodigo={} ORDER BY "data_iniSE" ASC;'.format(municipio),
                               conexao, index_col='id')
    df.set_index('data_iniSE', inplace=True)
    return df

def get_temperature_data(municipio=None):
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

def get_tweet_data(municipio=None):
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
