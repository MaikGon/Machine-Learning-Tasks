from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_decision_regions
from sklearn import metrics
from sklearn.cluster import KMeans, MeanShift, AffinityPropagation, OPTICS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import cv2 # remove when using plt


def compare_clusterization_methods():
    data = datasets.load_iris()
    X, y = data.data, data.target

    kmeans = KMeans(n_clusters=3, algorithm='elkan').fit(X)
    meanshift = MeanShift().fit(X)
    affinity = AffinityPropagation().fit(X)
    opt = OPTICS().fit(X)

    print(y)
    print(kmeans.labels_)
    print(kmeans.cluster_centers_)

    acc_kmeans = metrics.adjusted_rand_score(y, kmeans.labels_)
    print("Kmeans accuracy rand: ", acc_kmeans)
    acc_kmeans = metrics.adjusted_mutual_info_score(y, kmeans.labels_)
    print("Kmeans accuracy mutual: ", acc_kmeans)

    acc_meanshift = metrics.adjusted_rand_score(y, meanshift.labels_)
    print("Meanshift accuracy rand: ", acc_meanshift)
    acc_meanshift = metrics.adjusted_mutual_info_score(y, meanshift.labels_)
    print("Meanshift accuracy mutual: ", acc_meanshift)

    acc_affinity = metrics.adjusted_rand_score(y, affinity.labels_)
    print("Affinity accuracy rand: ", acc_affinity)
    acc_affinity = metrics.adjusted_mutual_info_score(y, affinity.labels_)
    print("Affinity accuracy mutual: ", acc_affinity)

    acc_opt = metrics.adjusted_rand_score(y, opt.labels_)
    print("Opt accuracy rand: ", acc_opt)
    acc_opt = metrics.adjusted_mutual_info_score(y, opt.labels_)
    print("Opt accuracy mutual: ", acc_opt)
    acc_opt = metrics.calinski_harabasz_score(X, y)
    print("Opt accuracy Calinski: ", acc_opt)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)

    figs = plt.figure()
    axs = figs.add_subplot(111, projection='3d')
    axs.scatter(X[:, 0], X[:, 1], X[:, 2], c=kmeans.labels_)

    plt.show()


def elbow_kmeans():
    data = datasets.load_iris()
    X, y = data.data, data.target

    vals = []
    arr = []
    for i in range(2, 16):
        kmeans = KMeans(n_clusters=i, algorithm='elkan').fit(X)
        arr.append(i)
        vals.append(kmeans.inertia_)

    plt.plot(arr, vals, '-o')
    plt.show()


def dimensions():
    data = datasets.load_iris()
    X, y = data.data, data.target

    pca = PCA(n_components=2)
    pca.fit(X)
    pca = pca.transform(X)

    tse = TSNE(n_components=2)
    tse = tse.fit_transform(X)

    plt.figure()
    plt.plot(pca, y)
    plt.figure()
    plt.plot(tse, y)
    plt.show()


def beach_forest_classifier():
    images_arr = []
    y = []
    # path to dataset
    beach_dir = Path('./beach_forest/')

    # load images
    images = sorted([im_path for im_path in beach_dir.iterdir() if im_path.name.endswith('.jpeg')])

    # read images with openCV
    for im in sorted(images):
        if str(im)[-7] == 'f':
            y.append('Forest')
        elif str(im)[-7] == 'b':
            y.append('Beach')
        image = cv2.imread(str(im))
        images_arr.append(image)

    # colors histogram
    hist_arr = []
    for i in range(len(images_arr)):
        for j in range(2):
            hist = cv2.calcHist([images_arr[i]], [j], None, [8], [0, 256])
            hist_arr.append(hist)

    X = np.array(hist_arr)
    X = X.reshape(40, -1)

    kmeans = KMeans(n_clusters=2, algorithm='elkan', random_state=42).fit(X)

    dict1 = {0: 'Beach', 1: 'Forest'}
    results = []

    for i in range(len(kmeans.labels_)):
        results.append(dict1[kmeans.labels_[i]])

    print(results)
    print(y)

    acc_kmeans = metrics.adjusted_rand_score(y, results)
    print("Kmeans accuracy rand: ", acc_kmeans)
    acc_kmeans = metrics.adjusted_mutual_info_score(y, results)
    print("Kmeans accuracy mutual: ", acc_kmeans)


if __name__ == "__main__":
    # compare_clusterization_methods()
    # elbow_kmeans()
    # dimensions()
    beach_forest_classifier()
