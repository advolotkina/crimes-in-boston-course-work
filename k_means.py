from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import os.path
import numpy as np
import pickle
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics

def start_k_means(filename):
    df = pd.read_csv('./uploads/' + filename)
    print(df.values)
    objects_array = df.values
    objects_array = objects_array.tolist()
    objects = []
    for object in objects_array:
        objects.append(str(object).strip('[]'))

    location = df[["Lat", "Long"]]

    location = location.loc[(location["Lat"] > 40) & (location["Long"] < -60)]

    X = location.values

    #Нахождение оптимального количества кластеров
    cluster_range = range(1, 20)
    cluster_errors = []

    for num_clusters in cluster_range:
        clusters = KMeans(num_clusters)
        clusters.fit(X)
        cluster_errors.append(clusters.inertia_)

    clusters_df = pd.DataFrame({"num_clusters": cluster_range, "cluster_errors": cluster_errors})

    clusters_df[0:10]
    plt.figure()
    plt.plot(clusters_df.num_clusters, clusters_df.cluster_errors, marker="o")
    plt.savefig('./static/opt_clusters_num.png')

    km = KMeans(n_clusters=6)
    km.fit(X)
    y_km = km.predict(X)
    labels_array = km.labels_
    # labels_array.tolist()
    model_filename = "k_means_districts.pkl"
    pickle.dump(km, open("./models/"+model_filename, "wb"))
    #Оценка методом силуэтов
    score = metrics.silhouette_score(X, labels=km.labels_)
    f = open("./scores/"+"k_means_score", "w+")
    f.write(str(score))
    f.close()
    # score = 0.39
    centers = km.cluster_centers_

    # КЛастеризация по районам

    plt.figure(figsize=(40, 40))
    plt.scatter(X[:, 1], X[:, 0], c=y_km, s=50, cmap="viridis")
    plt.scatter(centers[:, 1], centers[:, 0], c="red", s=50, alpha=0.5)
    plt.title("Кластеризация по районам, кол-во кластеров = 6")
    plt.ylabel("Lat")
    plt.xlabel("Long")
    plt.savefig('./static/district_clustering.png')

    #Кластеризация по районам маленькая картинка для отчета
    plt.figure()
    plt.scatter(X[:, 1], X[:, 0], c=y_km, s=50, cmap="viridis")
    plt.scatter(centers[:, 1], centers[:, 0], c="red", s=50, alpha=0.5)
    plt.title("KMeans n_clusters = 6 in location")
    plt.ylabel("Lat")
    plt.xlabel("Long")
    plt.savefig('./static/district_clustering_review.png')

    #Кластеризация по районам и типам преступлений
    location = df[["Lat", "Long", "OFFENSE_CODE"]]

    location = location.loc[(location["Lat"] > 40) & (location["Long"] < -60)]
    X = location.values

    km = KMeans(n_clusters=6)
    km.fit(X)
    y_km = km.predict(X)
    model_filename2 = "k_means_districts_and_offence_codes.pkl"
    pickle.dump(km, open("./models/" + model_filename2, "wb"))
    #Оценка методом силуэтов
    score_2 = metrics.silhouette_score(X, labels=km.labels_)
    f = open("./scores/"+"k_means_score_2", "w+")
    f.write(str(score_2))
    f.close()
    # score_2 = 0.39
    fig = plt.figure(1, figsize=(40, 40))
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
    ax.scatter(X[:, 1], X[:, 0], X[:, 2], c=y_km, edgecolor="k", s=50)
    ax.set_xlabel("Long")
    ax.set_ylabel("Lat")
    ax.set_zlabel("OFFENSE CODE")
    plt.title("Кластеризация по районам и типам преступлений, кол-во кластеров = 6")
    plt.savefig('./static/district_and_offence_code_clustering_k_means.png')

    return score, score_2, objects, labels_array, model_filename, model_filename2

def predict_with_model(data_file,model_file1, model_file2):
    df = pd.read_csv('./uploads/' + data_file)
    objects_array = df.values
    objects_array = objects_array.tolist()
    objects = []
    for object in objects_array:
        objects.append(str(object).strip('[]'))
    location = df[["Lat", "Long"]]
    location = location.loc[(location["Lat"] > 40) & (location["Long"] < -60)]
    X = location.values

    km_districts = pickle.load(open("./uploads/"+model_file1, "rb"))
    labels_array = km_districts.labels_
    y_km = km_districts.predict(X)
    score = metrics.silhouette_score(X, labels=km_districts.labels_)
    f = open("./scores/" + "k_means_score", "w+")
    f.write(str(score))
    f.close()
    # score = 0.39
    centers = km_districts.cluster_centers_

    # КЛастеризация по районам

    plt.figure(figsize=(40, 40))
    plt.scatter(X[:, 1], X[:, 0], c=y_km, s=50, cmap="viridis")
    plt.scatter(centers[:, 1], centers[:, 0], c="red", s=50, alpha=0.5)
    plt.title("Кластеризация по районам, кол-во кластеров = 6")
    plt.ylabel("Lat")
    plt.xlabel("Long")
    plt.savefig('./static/district_clustering.png')

    # Кластеризация по районам маленькая картинка для отчета
    plt.figure()
    plt.scatter(X[:, 1], X[:, 0], c=y_km, s=50, cmap="viridis")
    plt.scatter(centers[:, 1], centers[:, 0], c="red", s=50, alpha=0.5)
    plt.title("KMeans n_clusters = 6 in location")
    plt.ylabel("Lat")
    plt.xlabel("Long")
    plt.savefig('./static/district_clustering_review.png')

    km_district_and_offence_code = pickle.load(open("./uploads/"+model_file2, "rb"))
    location = df[["Lat", "Long", "OFFENSE_CODE"]]
    location = location.loc[(location["Lat"] > 40) & (location["Long"] < -60)]
    X = location.values
    y_km = km_district_and_offence_code.predict(X)
    score_2 = metrics.silhouette_score(X, labels=km_district_and_offence_code.labels_)
    f = open("./scores/"+"k_means_score_2", "w+")
    f.write(str(score))
    f.close()
    # score_2 = 0.39
    print("second score "+str(score_2))
    fig = plt.figure(1, figsize=(40, 40))
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
    ax.scatter(X[:, 1], X[:, 0], X[:, 2], c=y_km, edgecolor="k", s=50)
    ax.set_xlabel("Long")
    ax.set_ylabel("Lat")
    ax.set_zlabel("OFFENSE CODE")
    plt.title("Кластеризация по районам и типам преступлений, кол-во кластеров = 6")
    plt.savefig('./static/district_and_offence_code_clustering_k_means.png')

    return score, score_2, objects, labels_array


def get_elements_and_labels(data_file):
    df = pd.read_csv('./uploads/' + data_file)
    objects_array = df.values
    objects_array = objects_array.tolist()
    objects = []
    for object in objects_array:
        objects.append(str(object).strip('[]'))
    km_districts = pickle.load(open("./uploads/" + "k_means_districts.pkl", "rb"))
    labels_array = km_districts.labels_
    dictionary = dict(zip(tuple(objects), tuple(labels_array)))
    return dictionary


if __name__ == '__main__':
    print('k-means clustering')
