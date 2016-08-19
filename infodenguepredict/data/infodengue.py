"""
This module performs data fetching from the Infodengue database.
for remote database access, we recomment establishing an SSH tunnel:
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
        df = pd.read_sql_query('select * from "Municipio"."Historico_alerta"',
                                conexao, index_col='id')
    else:
        df = pd.read_sql_query('select * from "Municipio"."Historico_alerta" where municipio_geocodigo={}'.format(municipio),
                               conexao, index_col='id')
    return df
