import pandas as pd
import os
import re
import holoviews as hv

from infodenguepredict.data.infodengue import get_cluster_data, get_city_names
from infodenguepredict.predict_settings import *

hv.extension('bokeh')


def cluster_viz(geocode, clusters, loginc=False):
    """
    Create a heatmap incidence  plot
    :param geocode: City whose cluster to plot
    :param clusters: clusters from State
    :param loginc: whether to use log y scale for incidence
    :return: Holoviews plot
    """
    data, group = get_cluster_data(geocode=geocode, clusters=clusters,
                                   data_types=DATA_TYPES, cols=['casos'])

    city_names = dict(get_city_names(group))
    df_hm = data.reset_index().rename(columns={'index': 'week'})
    df_hm = pd.melt(df_hm, id_vars=['week'], var_name='city', value_name='incidence')
    df_hm['city'] = [int(re.sub('casos_', '', i)) for i in df_hm.city]
    df_hm['city'] = [city_names[i] for i in df_hm.city]

    #     return df_hm
    curve_opts = dict(line_width=10, line_alpha=0.4, tools=[], logy=loginc)
    overlay_opts = dict(width=900, height=200, tools=[])
    hm_opts = dict(width=900, height=500, tools=[], logz=True, invert_yaxis=False, xrotation=90,
                   labelled=[], toolbar=None, xaxis=None)

    heatmap = hv.HeatMap(df_hm)
    heatmap.toolbar_location = None
    graphs = [hv.Curve((data.index, data[i]), 'Time', 'Incidence') for i in data.columns]
    final = graphs[0]
    for i in graphs[1:]:
        final = final * i

    opts = {'HeatMap': {'plot': hm_opts}, 'Overlay': {'plot': overlay_opts},
            'Curve': {'plot': curve_opts,
                      'style': dict(color='blue', line_alpha=0.2)}}
    return (heatmap + final).opts(opts).cols(1)


if __name__ == "__main__":
    renderer = hv.Store.renderers['bokeh']
    renderer.dpi = 600
    if not os.path.exists('cluster_figs'):
        os.mkdir('cluster_figs')
    for STATE in ['ES']:
        clusters = pd.read_pickle('clusters_{}.pkl'.format(STATE))

        for c in clusters:
            if len(c) == 1:
                continue
            plot = cluster_viz(c[0], clusters)
            renderer.save(plot, 'cluster_figs/cluster_{}'.format(str(c[0])), 'png')
