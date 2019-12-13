from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import os.path
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics

def start_k_means():
    df = pd.read_csv('./uploads/tmpex0j7dw9.csv')
    # Dataframe for clustering on location
    location = df[["Lat", "Long"]]
    # Drop NaN values
    location = location.dropna()
    # Add some restrictions
    location = location.loc[(location["Lat"] > 40) & (location["Long"] < -60)]
    # Convert dataframe to numpy array
    X = location.values
    # cluster_range = range(1, 20)
    # cluster_errors = []
    #
    # for num_clusters in cluster_range:
    #     clusters = KMeans(num_clusters)
    #     clusters.fit(X)
    #     cluster_errors.append(clusters.inertia_)
    #
    # clusters_df = pd.DataFrame({"num_clusters": cluster_range, "cluster_errors": cluster_errors})
    #
    # clusters_df[0:10]
    # plt.figure()
    # plt.plot(clusters_df.num_clusters, clusters_df.cluster_errors, marker="o");
    # plt.savefig('./static/opt_clusters_num.png')

    km = KMeans(n_clusters=6)
    km.fit(X)
    y_km = km.predict(X)
    labels = km.labels_

    print(labels)
    centers = km.cluster_centers_

    # Plot for clustering on location

    plt.figure(figsize=(40, 40))
    plt.scatter(X[:, 1], X[:, 0], c=y_km, s=50, cmap="viridis")
    plt.scatter(centers[:, 1], centers[:, 0], c="red", s=50, alpha=0.5)
    plt.title("Кластеризация по районам, кол-во кластеров = 6")
    plt.ylabel("Lat")
    plt.xlabel("Long")
    plt.savefig('./static/district_clustering.png')

    # plt.figure()
    # plt.scatter(X[:, 1], X[:, 0], c=y_km, s=50, cmap="viridis")
    # plt.scatter(centers[:, 1], centers[:, 0], c="red", s=50, alpha=0.5)
    # plt.title("KMeans n_clusters = 6 in location")
    # plt.ylabel("Lat")
    # plt.xlabel("Long")
    # plt.savefig('./static/district_clustering_review.png')

    location = df[["Lat", "Long", "OFFENSE_CODE"]]
    location = location.dropna()
    # Add some restrictions
    location = location.loc[(location["Lat"] > 40) & (location["Long"] < -60)]
    # Convert dataframe to numpy array
    X = location.values
    # Clustering with KMeans
    km = KMeans(n_clusters=3)
    km.fit(X)
    y_km = km.predict(X)
    labels = km.labels_
    centers = km.cluster_centers_
    # Plot for clustering on location and offense_code
    fig = plt.figure(1, figsize=(7, 7))
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
    ax.scatter(X[:, 1], X[:, 0], X[:, 2], c=y_km, edgecolor="k", s=50)
    ax.set_xlabel("Long")
    ax.set_ylabel("Lat")
    ax.set_zlabel("OFFENSE CODE")
    plt.title("KMeans n_clusters = 3 in location and offense_code")
    plt.savefig('./static/district_and_offence_code_clustering.png')



# def optimal_num_of_clusters():
#     if os.path.isfile('./static/opt_clusters_num.png'):
#         return
#     df = pd.read_csv('./uploads/tmpex0j7dw9.csv')
#     #Dataframe for clustering on location
#     location = df[["Lat","Long"]]
#     #Drop NaN values
#     location = location.dropna()
#     #Add some restrictions
#     location = location.loc[(location["Lat"] > 40) & (location["Long"] < -60)]
#     #Convert dataframe to numpy array
#     X = location.values
#     cluster_range = range( 1, 20 )
#     cluster_errors = []
#
#     for num_clusters in cluster_range:
#       clusters = KMeans( num_clusters )
#       clusters.fit( X )
#       cluster_errors.append( clusters.inertia_ )
#
#     clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )
#
#     clusters_df[0:10]
#     plt.figure()
#     plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" );
#     plt.savefig('./static/opt_clusters_num.png')

# def district_clustering():
#     if os.path.isfile('./static/district_clustering.png'):
#         return
#     df = pd.read_csv('./uploads/tmpex0j7dw9.csv')
#     #Dataframe for clustering on location
#     location = df[["Lat","Long"]]
#     #Drop NaN values
#     location = location.dropna()
#     #Add some restrictions
#     location = location.loc[(location["Lat"] > 40) & (location["Long"] < -60)]
#     #Convert dataframe to numpy array
#     X = location.values
#     #Clustering with KMeans
#     km = KMeans(n_clusters=6)
#     km.fit(X)
#     y_km = km.predict(X)
#     labels = km.labels_
#     print(labels)
#     centers = km.cluster_centers_
#
#     #Plot for clustering on location
#
#     plt.figure(figsize=(40,40))
#     plt.scatter(X[:,1], X[:,0], c=y_km, s=50, cmap="viridis")
#     plt.scatter(centers[:,1], centers[:,0], c="red", s=50, alpha=0.5)
#     plt.title("KMeans n_clusters = 6 in location")
#     plt.ylabel("Lat")
#     plt.xlabel("Long")
#     plt.savefig('./static/district_clustering.png')
#     if os.path.isfile('./static/district_clustering_review.png'):
#         return
#     plt.figure()
#     plt.scatter(X[:,1], X[:,0], c=y_km, s=50, cmap="viridis")
#     plt.scatter(centers[:,1], centers[:,0], c="red", s=50, alpha=0.5)
#     plt.title("KMeans n_clusters = 6 in location")
#     plt.ylabel("Lat")
#     plt.xlabel("Long")
#     plt.savefig('./static/district_clustering_review.png')

# def district_and_offence_code_clustering():
#     if os.path.isfile('./static/district_and_offence_code_clustering.png'):
#         return
#     df = pd.read_csv('./uploads/tmpex0j7dw9.csv')
#     #Dataframe for clustering on location and offense_code
#     location = df[["Lat","Long","OFFENSE_CODE"]]
#     location = location.dropna()
#     #Add some restrictions
#     location = location.loc[(location["Lat"] > 40) & (location["Long"] < -60)]
#     #Convert dataframe to numpy array
#     X = location.values
#     #Clustering with KMeans
#     km = KMeans(n_clusters=3)
#     km.fit(X)
#     y_km = km.predict(X)
#     labels = km.labels_
#     centers = km.cluster_centers_
#     #Plot for clustering on location and offense_code
#     fig = plt.figure(1, figsize=(7,7))
#     ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
#     ax.scatter(X[:,1], X[:,0], X[:,2], c=labels.astype(np.float), edgecolor="k", s=50)
#     ax.set_xlabel("Long")
#     ax.set_ylabel("Lat")
#     ax.set_zlabel("OFFENSE CODE")
#     plt.title("KMeans n_clusters = 3 in location and offense_code")
#     plt.savefig('./static/district_and_offence_code_clustering.png')

# def getScore():
#     df = pd.read_csv('./uploads/tmpex0j7dw9.csv')
#     # Dataframe for clustering on location
#     location = df[["Lat", "Long"]]
#     # Drop NaN values
#     location = location.dropna()
#     # Add some restrictions
#     location = location.loc[(location["Lat"] > 40) & (location["Long"] < -60)]
#     # Convert dataframe to numpy array
#     X = location.values
#     # Clustering with KMeans
#     km = KMeans(n_clusters=6)
#     km.fit(X)
#     y_km = km.predict(X)
#     score = metrics.silhouette_score(X, labels=km.labels_)
#     return score
#
if __name__ == '__main__':
    print('main')
