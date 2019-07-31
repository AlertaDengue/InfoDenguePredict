import pickle
import pandas as pd
import seaborn as sns
import scipy.cluster.hierarchy as hac
import matplotlib.pyplot as plt

from scipy.spatial import distance as ssd
from infodenguepredict.analysis.distance import distance, alocate_data
from infodenguepredict.data.infodengue import get_city_names
from infodenguepredict.predict_settings import *


def hierarchical_clustering(df, t, method='complete'):
    """
    :param method: Clustering method
    :param df: Triangular distances matrix
    :return:
    """
    Z = hac.linkage(ssd.squareform(df.values.T + df.values), method=method)

    ind = hac.fcluster(Z, t * max(Z[:, 2]), 'distance')
    grouped = pd.DataFrame(list(zip(ind, df.index))).groupby(0)
    clusters = [group[1][1].values for group in grouped]
    return Z, clusters


def matrix_cluster(cities_list, clusters):
    df = pd.DataFrame(index=cities_list, columns=['cluster'])
    for pos, cluster in enumerate(clusters):
        df.loc[cluster] = pos

    df.to_csv('list_cluster_{}.csv'.format(STATE))
    return 'List of clusters csv saved'


def create_cluster(state, cols, t):
    cities_list = alocate_data(state)
    dists = distance(cities_list, cols)

    dists_full = dists + dists.T
    sns_plot = sns.clustermap(dists_full, cmap="vlag")
    sns_plot.savefig("cluster_corr_{}.png".format(state), dpi=400)

    Z, clusters = hierarchical_clustering(dists, t=t)
    print(clusters)
    matrix_cluster(cities_list=cities_list, clusters=clusters)

    with open('clusters_{}.pkl'.format(state), 'wb') as fp:
        pickle.dump(clusters, fp)
    print("{} clusters saved".format(state))
    name_ind = get_city_names(list(dists.index))

    return Z, name_ind


def llf(id):
    return name_ind[id][1]


if __name__ == "__main__":

    for STATE in ['CE']:#['PR', 'CE', 'ES']:
        Z, name_ind = create_cluster(STATE, CLUSTER_VARS, COLOR_THRESHOLD)

        plt.figure(figsize=(10, 25))
        # plt.title('Hierarchical Clustering Dendrogram')
        # plt.xlabel('sample index')
        # plt.ylabel('distance')
        # plt.tight_layout()
        hac.dendrogram(
            Z,
            orientation='right',
            # leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8,  # font size for the x axis labels
            leaf_label_func=llf,
            color_threshold=COLOR_THRESHOLD * max(Z[:, 2])
        )

        plt.savefig('{}/cluster{}_{}.png'.format('../models/saved_models', STATE, COLOR_THRESHOLD), dpi=300, bbox_inches='tight')

        plt.show()


