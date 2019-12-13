import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import *
from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.cluster.hierarchy import fcluster


# def load_data():
#     df = pd.read_csv('./uploads/tmpex0j7dw9.csv')

def start_hierarhy():
    print("HI")
    df = pd.read_csv('./uploads/tmpex0j7dw9.csv')
    location = df[["Lat", "Long"]]
    location = location.loc[(location["Lat"] > 40) & (location["Long"] < -60)]
    location = location.dropna()
    location = location.head(10000)
    data_dist = pdist(location)
    data_linkage = linkage(data_dist, method='complete')
    last = data_linkage[-10:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)
    plt.plot(idxs, last_rev)
    acceleration = np.diff(last, 2)
    acceleration_rev = acceleration[::-1]
    plt.plot(idxs[:-2] + 1, acceleration_rev)
    plt.savefig('./static/hierarchy_clusters.png')
    k = acceleration_rev.argmax() + 2
    print("clusters:", k)
    clusters = fcluster(data_linkage, 50, criterion='distance')
    clusters = fcluster(data_linkage, 5, criterion='maxclust')
    plt.figure(figsize=(10, 8))
    plt.scatter(location.iloc[:, 1], location.iloc[:, 0], c=clusters, cmap='flag')
    plt.savefig('./static/hierarchy_clustering.png')


if __name__ == '__main__':
    start_hierarhy()