import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from decouple import config


from infodenguepredict.data.infodengue import get_cluster_data, get_city_names
from infodenguepredict.models.random_forest import build_lagged_features


def loss_colormap(state, models, metric='mean_squared_error', predict_n=1):
    """
    Colormap viz for model losses.
    :param state: State to plot
    :param models: List of models to show -> ['lstm', 'rf', 'tpot']
    :param metric: Metric for y-axis -> ['mean_absolute_error', 'explained_variance_score', 'mean_squared_error',
       'mean_squared_log_error', 'median_absolute_error', 'r2_score']
    :param predict_n: Which window to compare
    :return: Plot
    """

    clusters = pd.read_pickle('../../analysis/clusters_{}.pkl'.format(state))
    clusters = [y for x in clusters for y in x]
    df = pd.DataFrame(columns=models, index=clusters)
    for city in clusters:
        if 'rf' in models:
            rf = pd.read_pickle('../saved_models/random_forest/{}/rf_metrics_{}.pkl'.format(state, city))
            df['rf'][city] = rf[predict_n][metric]
        if 'lstm' in models:
            lstm = pd.read_pickle('../saved_models/lstm/{}/lstm_metrics_{}.pkl'.format(state, city))
            df['lstm'][city] = lstm[predict_n][metric]
        if 'tpot' in models:
            tpot = pd.read_pickle('../saved_models/tpot/{}/tpot_metrics_{}.pkl'.format(state, city))
            df['tpot'][city] = tpot[predict_n][metric]
        if 'rqf' in models:
            rqf = pd.read_pickle('../saved_models/quantile_forest/{}/qf_metrics_{}.pkl'.format(state, city))
            df['rqf'][city] = rqf[predict_n][metric]

    df = df[df.columns].astype('float')
    # falta normalizar a data?
    sns_plot = sns.heatmap(df, cmap='vlag')
    plt.savefig('{}_losses_heatmap.png'.format(state), dpi=400)
    plt.show()

    return None


def calculate_mape(state, lookback, horizon):
    clusters = pd.read_pickle('../analysis/clusters_{}.pkl'.format(state))

    for cluster in clusters:
        data_full, group = get_cluster_data(geocode=cluster[0], clusters=clusters,
                                            data_types=['alerta'], cols=['casos_est', 'casos'])
        for city in cluster:
            print(city)

            target = 'casos_est_{}'.format(city)
            casos_est_columns = ['casos_est_{}'.format(i) for i in group]
            casos_columns = ['casos_{}'.format(i) for i in group]

            data = data_full.drop(casos_columns, axis=1)
            data_lag = build_lagged_features(data, lookback)
            data_lag.dropna()
            targets = {}
            for d in range(1, horizon + 1):
                if d == 1:
                    targets[d] = data_lag[target].shift(-(d - 1))
                else:
                    targets[d] = data_lag[target].shift(-(d - 1))[:-(d - 1)]

            X_data = data_lag.drop(casos_est_columns, axis=1)
            X_train, X_test, y_train, y_test = train_test_split(X_data, data_lag[target],
                                                                train_size=0.7, test_size=0.3, shuffle=False)

            try:
                metrics = pd.read_pickle('~/Documentos/resultados_infodengue/lasso/{}/lasso_metrics_{}.pkl'.format(state, city))
            except EOFError:
                print('---------------------------------')
                print('ERROR', 'eof', city)
                print('----------------------------------')

            if metrics.shape[1] != 4:
                print('---------------------------------')
                print('ERROR', 'shape', city)
                print('----------------------------------')
                continue

            values = []
            for d in range(1, horizon + 1):
                mae = metrics[d]['mean_absolute_error']
                tgtt = targets[d][len(X_train):]

                factor = (len(tgtt) / (len(tgtt) - 1)) * sum([abs(i - (tgtt[pos])) for pos, i in enumerate(tgtt[1:])])
                if factor == 0:
                    values.append(np.nan)
                else:
                    values.append(mae / factor)

            metrics.loc['mean_absolute_scaled_error'] = values
            metrics.to_pickle('~/Documentos/resultados_infodengue/lasso/{}/lasso_metrics_{}.pkl'.format(state, city))
    return None


def loss_scatter(state, models, metric='mean_squared_error', predict_n=1):
    """
    Scatter viz for model losses.
    :param state: State to plot
    :param models: List of models to show -> ['lstm', 'rf', 'tpot']
    :param xaxis: List o xaxis possibilites -> ['cluster_size', 'pop_size', 'total_cases', 'latitude']
    :param metric: Metric for y-axis -> ['mean_absolute_error', 'explained_variance_score', 'mean_squared_error',
       'mean_squared_log_error', 'median_absolute_error', 'r2_score']
    :param predict_n: Which window to compare
    :return: Plot
    """
    conexao = create_engine("postgresql://{}:{}@{}/{}".format(config('PSQL_USER'),
                                                              config('PSQL_PASSWORD'),
                                                              config('PSQL_HOST'),
                                                              config('PSQL_DB')))
    if state == 'CE':
        s = 'CE'
    if state == 'RJ':
        s = 'Rio de Janeiro'
    if state == 'PR':
        s = 'Paraná'

    sql = 'select geocodigo,nome,populacao,casos_est from "Dengue_global"."Municipio" m JOIN "Municipio"."Historico_alerta" h ON m.geocodigo=h.municipio_geocodigo where uf=\'{}\';'.format(
        s)
    data = pd.read_sql_query(sql, conexao)
    grouped = data.groupby('geocodigo')
    clusters = pd.read_pickle('infodenguepredict/analysis/clusters_{}.pkl'.format(state))
    cities = [y for x in clusters for y in x]

    df = pd.DataFrame(columns=models + ['cluster_size', 'pop_size', 'total_casos', 'n_epidemia'], index=cities)
    for city in cities:
        group = grouped.get_group(city)
        df['total_casos'][city] = group['casos_est'].sum()
        city_pop = group['populacao'].iloc[0]
        df['pop_size'][city] = city_pop
        df['n_epidemia'][city] = len(group[group['casos_est'] > int(city_pop / 1000)])

        if 'rf' in models:
            rf = pd.read_pickle('~Documentos/resultados_infodengue/random_forest/{}/rf_metrics_{}.pkl'.format(state, city))
            df['rf'][city] = rf[predict_n][metric]
        if 'lstm' in models:
            lstm = pd.read_pickle('~Documentos/resultados_infodengue/lstm/{}/metrics_lstm_{}.pkl'.format(state, city))
            df['lstm'][city] = lstm[predict_n][metric]
        if 'lasso' in models:
            lasso = pd.read_pickle('~Documentos/resultados_infodengue/lasso/{}/lasso_metrics_{}.pkl'.format(state, city))
            try:
                df['lasso'][city] = lasso[predict_n][metric]
            except KeyError:
                df['lasso'][city] = np.nan

    for cluster in clusters:
        df['cluster_size'].loc[cluster] = len(cluster)
    df = df[df.columns].astype('float')
    #     df = df[df.total_casos > 100]

    df = df[df.pop_size < df.pop_size.mean() + 4 * df.pop_size.std()]
    df = df[df.total_casos < df.total_casos.mean() + 4 * df.total_casos.std()]

    fig, axs = plt.subplots(1, 3, figsize=(30, 7))
    colors = ['b', 'r', 'g']

    for pos, m in enumerate(models):
        df_m = df[df[m] < df[m].mean() + 1 * df[m].std()]
        df_m.plot.scatter(x='total_casos', y=m, ax=axs[0], alpha=0.3, grid=True, c=colors[pos])
        df_m.plot.scatter(x='pop_size', y=m, ax=axs[1], alpha=0.3, grid=True, c=colors[pos])
        df_m.plot.scatter(x='n_epidemia', y=m, ax=axs[2], alpha=0.3, grid=True, c=colors[pos])

        plt.legend(models)

    return df


if __name__ == "__main__":
    loss_colormap('RJ', ['rqf'])
