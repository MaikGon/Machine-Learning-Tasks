from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from matplotlib.colors import ListedColormap
from mlxtend.plotting import plot_decision_regions
from yellowbrick.contrib.classifier import DecisionViz


def lab3_1():
    data = datasets.load_iris()
    X = data.data[:, [0, 2]]
    y = data.target
    X = StandardScaler().fit_transform(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.8, random_state=42, stratify=y)

    # scaler = StandardScaler().fit_transform(X_train, y_train)
    # scaler = preprocessing.scale(X_train)
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.axvline(x=0)
    # plt.axhline(y=0)
    # plt.title('Iris sepal features')
    # plt.xlabel('sepal length (cm)')
    # plt.ylabel('sepal width (cm)')
    # plt.show()

    classifier = svm.SVC()
    classifier.fit(X_train, y_train)

    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot()
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')

    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
               edgecolors='k', alpha=0.6)

    # plot_decision_regions(X=X, y=y, clf=classifier, legend=2)

    viz = DecisionViz(classifier, X[:2], y[:2], features=data.feature_names[:2],
                      classes=list(data.target_names[:2]))
    viz.fit(X_train, y_train)
    viz.fit_draw_show(X_test, y_test)

    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.title('SVM on Iris')
    plt.show()


def lab3_1_1():
    data = datasets.load_iris()
    X, y = data.data, data.target

    print(np.bincount(y)) # ilosc binow (cech)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.8, random_state=42, stratify=y)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Iris sepal features')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')

    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(X_train)
    min_max_scaler = min_max_scaler.transform(X)

    plt.figure()
    plt.scatter(min_max_scaler[:, 0], min_max_scaler[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Iris sepal features')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')

    st_scaler = preprocessing.StandardScaler()
    st_scaler.fit(X_train)
    st_scaler = st_scaler.transform(X)

    plt.figure()
    plt.scatter(st_scaler[:, 0], st_scaler[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Iris sepal features')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')
    plt.show()


if __name__ == "__main__":
    # lab3_1()
    lab3_1_1()
