import pickle
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as hac
from scipy.spatial import distance as ssd
import matplotlib.pyplot as plt
from infodenguepredict.analysis.distance import distance, alocate_data
from infodenguepredict.data.infodengue import get_city_names
from functools import lru_cache

def hierarchical_clustering(df, t, method='average'):
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

@lru_cache(maxsize=None)
def create_cluster(state, t):
    cities_list = alocate_data(state)
    dists = distance(cities_list)
    Z, clusters = hierarchical_clustering(dists, t=t)

    with open('clusters_{}.pkl'.format(state), 'wb') as fp:
        pickle.dump(clusters, fp)

    print("{} clusters saved".format(state))
    name_ind = get_city_names(list(dists.index))
    return Z, name_ind

def llf(id):
    return name_ind[id][1]

if __name__ == "__main__":
    # for t in np.arange(0.6):
    #     print(t)
    t = 0.6
    Z, name_ind = create_cluster("RJ", t)

    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    hac.dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        leaf_label_func=llf,
        color_threshold=t * max(Z[:, 2])
    )
    plt.savefig('cluster_{}.png'.format(t))
    plt.show()
