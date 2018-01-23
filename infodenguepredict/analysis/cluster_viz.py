import pandas as pd
import pickle
import re
import holoviews as hv

from infodenguepredict.data.infodengue import  get_cluster_data, get_city_names
from infodenguepredict.predict_settings import *

hv.extension('bokeh')


def cluster_viz(geocode, clusters):
    data, group = get_cluster_data(geocode=geocode, clusters=clusters,
                                   data_types=DATA_TYPES, cols=['casos'])

    city_names = dict(get_city_names(group))
    df_hm = data.reset_index().rename(columns={'index': 'week'})
    df_hm = pd.melt(df_hm, id_vars=['week'], var_name='city', value_name='incidence')
    df_hm['city'] = [re.sub('casos_', '', i) for i in df_hm.city]
    df_hm['city'] = [city_names[int(i)] for i in df_hm.city]

    curve_opts = dict(line_width=10, line_alpha=0.4)
    overlay_opts = dict(width=900, height=200)
    hm_opts = dict(width=900, height=500, tools=[None], logz=True, invert_yaxis=True, xrotation=90,
                   labelled=[], toolbar=None, xaxis=None)

    heatmap = hv.HeatMap(df_hm, label='Dengue Incidence')
    graphs = [hv.Curve((data.index, data[i]), 'Time', 'Incidence') for i in data.columns]
    final = graphs[0]
    for i in graphs[1:]:
        final = final * i

    opts = {'HeatMap': {'plot': hm_opts}, 'Overlay': {'plot': overlay_opts}, 'Curve': {'plot': curve_opts}}
    return (heatmap + final).opts(opts).cols(1)


if __name__ == "__main__":
    state = 'RJ'
    renderer = hv.Store.renderers['bokeh']
    renderer.dpi = 600

    with open('clusters_{}.pkl'.format(state), 'rb') as fp:
        clusters = pickle.load(fp)

    for c in clusters:
        plot = cluster_viz(c[0], clusters)
        renderer.save(plot, '{}/cluster_{}'.format(FIG_PATH, str(c[0])), 'png')

