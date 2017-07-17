import numpy as np;
import seaborn as sns;
import pandas as pd
import re
from scipy import stats
import scipy.cluster.hierarchy as hac
from scipy.spatial import distance as ssd
import matplotlib.pyplot as plt
from infodenguepredict.data.infodengue import get_alerta_table, build_multicity_dataset
from infodenguepredict.analysis.distance import distance, alocate_data

def hierarchical_clustering(df, method='complete'):
    """
    :param method: Clustering method
    :param df: Dataframe with series as columns
    :return:
    """
    Z = hac.linkage(ssd.squareform(df.values.T + df.values), method=method)
    return Z


if __name__ == "__main__":
    cities_list = alocate_data("RJ")
    dists = distance(cities_list)
    Z = hierarchical_clustering(dists)
    # print(len(data.columns), Z.shape)

    dic = pd.read_excel('../data/codigos_rj.xlsx', names=['city','code'], header=None).set_index(
        'code')
    dic.index = dic.index.astype('str')
    codes_dict = dic.to_dict()['city']
    labels = [codes_dict[str(i)] for i in dists.columns]

    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    hac.dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        labels=labels,
    )
    plt.show()