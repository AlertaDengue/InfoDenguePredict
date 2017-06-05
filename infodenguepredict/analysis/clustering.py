import numpy as np;
import seaborn as sns;
import pandas as pd
from scipy import stats
import scipy.cluster.hierarchy as hac
import matplotlib.pyplot as plt
from infodenguepredict.data.infodengue import get_alerta_table, build_multicity_dataset


def hierarchical_clustering(df, method='complete'):
    """

    :param method: Clustering method
    :param df: Dataframe with series as columns
    :return:
    """
    Z = hac.linkage(df.values.T, method=method, metric='correlation')
    return Z


if __name__ == "__main__":
    data = build_multicity_dataset('RJ')
    data = data[[col for col in data.columns if col.startswith('casos') and not col.startswith('casos_est')]]
    Z = hierarchical_clustering(data)
    print(len(data.columns), Z.shape)
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    hac.dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        labels= data.columns,
    )
    plt.show()