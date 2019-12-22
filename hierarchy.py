import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import *
from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.cluster.hierarchy import fcluster
from sklearn import metrics
import pickle


# def load_data():
#     df = pd.read_csv('./uploads/tmpex0j7dw9.csv')

def start_hierarhy(filename):
    # df = pd.read_csv('./uploads/'+filename)
    df = pd.read_csv('./uploads/tmpex0j7dw9.csv')
    location = df[["Lat", "Long"]]
    location = location.loc[(location["Lat"] > 40) & (location["Long"] < -60)]
    location = location.dropna()
    location = location.head(9000)
    data_dist = pdist(location)
    data_linkage = linkage(data_dist, method='complete')
    last = data_linkage[-10:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)
    plt.plot(idxs, last_rev)
    acceleration = np.diff(last, 2)
    acceleration_rev = acceleration[::-1]
    plt.plot(idxs[:-2] + 1, acceleration_rev)
    plt.savefig('./static/hierarchy_clustersNEW1.png')
    k = acceleration_rev.argmax() + 4
    print("clusters:", k)
    clusters = fcluster(data_linkage, 50, criterion='distance')
    clusters = fcluster(data_linkage, 4, criterion='maxclust')
    plt.figure(figsize=(10, 8))
    plt.scatter(location.iloc[:, 1], location.iloc[:, 0], c=clusters, cmap='flag')
    plt.savefig('./static/hierarchy_clusteringNEW1.png')
    score = metrics.silhouette_score(location, clusters)
    model_filename = "hierarhy_districts_and_offence_codes.pkl"
    pickle.dump(clusters, open(model_filename, "wb"))
    location = df[["Lat", "Long", "OFFENSE_CODE"]]
    location = location.loc[(location["Lat"] > 40) & (location["Long"] < -60)]
    location = location.dropna()
    location = location.head(8000)
    data_dist = pdist(location)
    data_linkage = linkage(data_dist, method='complete')
    clusters = fcluster(data_linkage, 50, criterion='distance')
    clusters = fcluster(data_linkage, 4, criterion='maxclust')
    fig = plt.figure(1, figsize=(40, 40))
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
    ax.scatter(location.iloc[:, 1], location.iloc[:, 0], location.iloc[:, 2], c=clusters, edgecolor="k", s=50)
    ax.set_xlabel("Long")
    ax.set_ylabel("Lat")
    ax.set_zlabel("OFFENSE CODE")
    plt.title("Hierarchy in location and offense_code")
    plt.savefig('./static/district_and_offence_code_clustering.png')
    score2 = metrics.silhouette_score(location, clusters)
    model_filename2 = "hierarhy_districts_and_offence_codes.pkl"
    pickle.dump(clusters, open(model_filename2, "wb"))
    return score,score2,model_filename,model_filename2

def predict_with_model(data_file,model_file1, model_file2):
    df = pd.read_csv('./uploads/' + data_file)

    location = df[["Lat", "Long"]]
    location = location.loc[(location["Lat"] > 40) & (location["Long"] < -60)]
    location = location.dropna()
    location = location.head(9000)
    hierarhy_districts = pickle.load(open("./uploads/"+model_file1, "rb"))
    score = metrics.silhouette_score(location, hierarhy_districts)
    f = open("./scores/" + "hierarhy_score", "w+")
    f.write(str(score))
    f.close()

    # КЛастеризация по районам
    plt.figure(figsize=(10, 8))
    plt.scatter(location.iloc[:, 1], location.iloc[:, 0], c=hierarhy_districts, cmap='flag')
    plt.savefig('./static/hierarchy_clusteringNEW1.png')

    hierarhy_district_and_offence_code = pickle.load(open("./uploads/"+model_file2, "rb"))
    location = location.loc[(location["Lat"] > 40) & (location["Long"] < -60)]
    location = location.dropna()
    location = location.head(9000)

    score_2 = metrics.silhouette_score(location, hierarhy_district_and_offence_code)
    f = open("./scores/" + "hierarhy_score_2", "w+")
    f.write(str(score))
    f.close()

    fig = plt.figure(1, figsize=(40, 40))
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
    ax.scatter(location.iloc[:, 1], location.iloc[:, 0], location.iloc[:, 2], c=hierarhy_district_and_offence_code, edgecolor="k", s=50)
    ax.set_xlabel("Long")
    ax.set_ylabel("Lat")
    ax.set_zlabel("OFFENSE CODE")
    plt.title("Hierarchy in location and offense_code")
    plt.savefig('./static/district_and_offence_code_clustering.png')

    return score, score_2,


if __name__ == '__main__':
    start_hierarhy()