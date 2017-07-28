import pickle
import pandas as pd
import scipy.cluster.hierarchy as hac
from scipy.spatial import distance as ssd
import matplotlib.pyplot as plt
from infodenguepredict.analysis.distance import distance, alocate_data

def hierarchical_clustering(df, method='complete'):
    """
    :param method: Clustering method
    :param df: Triangular distances matrix
    :return:
    """
    Z = hac.linkage(ssd.squareform(df.values.T + df.values), method=method)

    ind = hac.fcluster(Z, 0.7 * max(Z[:, 2]), 'distance')
    grouped = pd.DataFrame(list(zip(ind, df.index))).groupby(0)
    clusters = [group[1][1].values for group in grouped]
    return Z, clusters


def create_cluster(state):
    cities_list = alocate_data("RJ")
    dists = distance(cities_list)
    Z, clusters = hierarchical_clustering(dists)

    with open('clusters_{}.pkl'.format('RJ'), 'wb') as fp:
        pickle.dump(clusters, fp)

    print("{} clusters saved".format(state))
    return Z

if __name__ == "__main__":
    Z = create_cluster("RJ")

    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    hac.dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        # labels=labels,
    )