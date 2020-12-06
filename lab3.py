from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from matplotlib.colors import ListedColormap
from mlxtend.plotting import plot_decision_regions
from yellowbrick.contrib.classifier import DecisionViz
from sklearn import svm, tree, ensemble, linear_model
from sklearn import metrics
from sklearn import model_selection
import seaborn as sns
import pandas as pd


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

    print(np.bincount(y))  # ilosc binow (cech)

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


def lab3_2():
    data = datasets.load_iris()
    X, y = data.data, data.target
    X = X[:, [0, 2]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42, stratify=y)
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(X_train)

    X_train = min_max_scaler.transform(X_train)
    X_test = min_max_scaler.transform(X_test)

    svm_clf = svm.SVC(kernel='poly')
    svm_clf.fit(X_train, y_train)
    acc_svm = metrics.accuracy_score(y_test, svm_clf.predict(X_test))
    print("SVM accuracy: ", acc_svm)

    linear_clf = linear_model.LogisticRegression()
    linear_clf.fit(X_train, y_train)
    acc_lin = metrics.accuracy_score(y_test, linear_clf.predict(X_test))
    print("Logistic regression accuracy: ", acc_lin)

    tree_clf = tree.DecisionTreeClassifier(max_depth=5)
    tree_clf.fit(X_train, y_train)
    acc_tree = metrics.accuracy_score(y_test, tree_clf.predict(X_test))
    print("Decision tree accuracy: ", acc_tree)

    forest_clf = ensemble.RandomForestClassifier(n_estimators=500)
    forest_clf.fit(X_train, y_train)
    acc_for = metrics.accuracy_score(y_test, forest_clf.predict(X_test))
    print("Random forest accuracy: ", acc_for)

    # Now search for best hyperparameters
    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
    ]

    clf_gs = model_selection.GridSearchCV(estimator=svm.SVC(), param_grid=param_grid, n_jobs=4, verbose=20)
    clf_gs.fit(X_train, y_train)
    print(clf_gs.cv_results_)

    pvt = pd.pivot_table(pd.DataFrame(clf_gs.cv_results_), values='mean_test_score',
        index='param_kernel', columns='param_C')

    sns.heatmap(pvt)
    plt.show()

    print("SVM: ", svm_clf.predict(min_max_scaler.transform([[8.0, 4.0]])))
    print("Linear: ", linear_clf.predict(min_max_scaler.transform([[8.0, 4.0]])))
    print("Decision tree: ", tree_clf.predict(min_max_scaler.transform([[8.0, 4.0]])))
    print("Random forest: ", forest_clf.predict(min_max_scaler.transform([[8.0, 4.0]])))

    plt.figure()
    plot_decision_regions(X_train, y_train, clf=svm_clf, legend=2)
    plt.figure()
    plot_decision_regions(X_train, y_train, clf=linear_clf, legend=2)
    plt.figure()
    plot_decision_regions(X_train, y_train, clf=tree_clf, legend=2)
    plt.figure()
    plot_decision_regions(X_train, y_train, clf=forest_clf, legend=2)
    plt.show()


def lab3_3():
    # fashion mnist, search for best hyperparameters
    data = datasets.fetch_openml(name='Fashion-MNIST')
    X = data['data']
    X = X[:, [0, 2]]
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42, stratify=y)
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(X_train)

    # Now search for best hyperparameters
    param_grid = [
        {'C': [1, 10, 100], 'kernel': ['linear']},
        {'C': [1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
    ]

    clf_gs = model_selection.GridSearchCV(estimator=svm.SVC(), param_grid=param_grid, n_jobs=4, verbose=20)
    clf_gs.fit(X_train, y_train)
    print(clf_gs.cv_results_)

    pvt = pd.pivot_table(pd.DataFrame(clf_gs.cv_results_), values='mean_test_score',
        index='param_kernel', columns='param_C')

    sns.heatmap(pvt)
    plt.show()

    svm_clf = svm.SVC(kernel='poly')
    svm_clf.fit(X_train, y_train)
    acc_svm = metrics.accuracy_score(y_test, svm_clf.predict(X_test))
    print("SVM accuracy: ", acc_svm)

    plt.figure()
    plot_decision_regions(X_train, y_train, clf=svm_clf, legend=2)
    plt.show()


if __name__ == "__main__":
    # lab3_1()
    # lab3_1_1()
    lab3_2()
    # lab3_3()
