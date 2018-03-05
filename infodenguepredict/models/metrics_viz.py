import pandas as pd
import seaborn as sns
import numpy as np


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

    clusters = pd.read_pickle('infodenguepredict/analysis/clusters_{}.pkl'.format(state))
    clusters = [y for x in clusters for y in x]
    df = pd.DataFrame(columns=models, index=clusters)
    for city in clusters:
        if 'rf' in models:
            rf = pd.read_pickle('../resultados_infodengue/random_forest/{}/rf_metrics_{}.pkl'.format(state, city))
            df['rf'][city] = rf[predict_n][metric]
        if 'lstm' in models:
            lstm = pd.read_pickle('../resultados_infodengue/lstm/{}/lstm_metrics_{}.pkl'.format(state, city))
            df['lstm'][city] = lstm[predict_n][metric]
        if 'tpot' in models:
            tpot = pd.read_pickle('../resultados_infodengue/tpot/{}/tpot_metrics_{}.pkl'.format(state, city))
            df['tpot'][city] = tpot[predict_n][metric]

    df = df[df.columns].astype('float')
    # falta normalizar a data?
    sns_plot = sns.heatmap(df, cmap='vlag')
    sns_plot.savefig('{}_losses_heatmap.png', dpi=400)

    return None

