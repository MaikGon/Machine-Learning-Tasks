from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn import svm, tree
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from scipy.stats import zscore


def ensamble():
    df = sns.load_dataset('anscombe')
    print(df)
    X1, X2, X3, X4 = [], [], [], []
    y1, y2, y3, y4 = [], [], [], []

    for i in range(len(df)):
        if df['dataset'][i] == 'I':
            X1.append(df['x'][i])
            y1.append(df['y'][i])
        elif df['dataset'][i] == 'II':
            X2.append(df['x'][i])
            y2.append(df['y'][i])
        elif df['dataset'][i] == 'III':
            X3.append(df['x'][i])
            y3.append(df['y'][i])
        else:
            X4.append(df['x'][i])
            y4.append(df['y'][i])

    # Show the results
    sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=df, col_wrap=2,
               ci=None, palette="muted", height=4, scatter_kws={"s": 50, "alpha": 1})

    forest_clf_1 = RandomForestClassifier(n_estimators=500)
    forest_clf_2 = RandomForestClassifier(n_estimators=500)
    forest_clf_3 = RandomForestClassifier(n_estimators=500)
    forest_clf_4 = RandomForestClassifier(n_estimators=500)

    X1 = np.array(X1)
    X1 = X1.reshape(11, -1)
    y1 = np.array(y1, dtype=np.int64)

    forest_clf_1.fit(X1, y1)
    y_predicted = []

    for data in X1:
        y_predicted.append(forest_clf_1.predict([data]))

    plt.figure()
    plt.scatter(X1, y1)
    plt.scatter(X1, y_predicted, marker="*", c='red')
    plt.show()


def missing_data():
    data = datasets.fetch_openml(name='diabetes')
    X = data['data']
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42, stratify=y)

    simple_imp = SimpleImputer()
    simple_imp.fit(X_train, y_train)

    iter_imp = IterativeImputer()
    iter_imp.fit(X_train, y_train)

    knn_imp = KNNImputer()
    knn_imp.fit(X_train, y_train)

    forest_clf = RandomForestClassifier()
    forest_clf.fit(X_train, y_train)
    acc_forest = metrics.accuracy_score(y_test, forest_clf.predict(X_test))
    print("Random forest accuracy: ", acc_forest)

    forest_clf.fit(simple_imp.transform(X_train), y_train)
    acc_svm = metrics.accuracy_score(y_test, forest_clf.predict(simple_imp.transform(X_test)))
    print("Random forest accuracy simple: ", acc_svm)

    forest_clf.fit(iter_imp.transform(X_train), y_train)
    acc_svm = metrics.accuracy_score(y_test, forest_clf.predict(iter_imp.transform(X_test)))
    print("Random forest accuracy iterative: ", acc_svm)

    forest_clf.fit(knn_imp.transform(X_train), y_train)
    acc_svm = metrics.accuracy_score(y_test, forest_clf.predict(knn_imp.transform(X_test)))
    print("Random forest accuracy knn: ", acc_svm)

    tree_clf = tree.DecisionTreeClassifier()
    tree_clf.fit(X_train, y_train)
    acc_tree = metrics.accuracy_score(y_test, tree_clf.predict(X_test))
    print("Decision tree accuracy: ", acc_tree)

    tree_clf.fit(simple_imp.transform(X_train), y_train)
    acc_svm = metrics.accuracy_score(y_test, tree_clf.predict(simple_imp.transform(X_test)))
    print("Decision tree accuracy simple: ", acc_svm)

    tree_clf.fit(iter_imp.transform(X_train), y_train)
    acc_svm = metrics.accuracy_score(y_test, tree_clf.predict(iter_imp.transform(X_test)))
    print("Decision tree accuracy iterative: ", acc_svm)

    tree_clf.fit(knn_imp.transform(X_train), y_train)
    acc_svm = metrics.accuracy_score(y_test, tree_clf.predict(knn_imp.transform(X_test)))
    print("Decision tree accuracy knn: ", acc_svm)

    svm_clf = svm.SVC()
    svm_clf.fit(X_train, y_train)
    acc_svm = metrics.accuracy_score(y_test, svm_clf.predict(X_test))
    print("SVM accuracy: ", acc_svm)

    svm_clf.fit(simple_imp.transform(X_train), y_train)
    acc_svm = metrics.accuracy_score(y_test, svm_clf.predict(simple_imp.transform(X_test)))
    print("SVM accuracy simple: ", acc_svm)

    svm_clf.fit(iter_imp.transform(X_train), y_train)
    acc_svm = metrics.accuracy_score(y_test, svm_clf.predict(iter_imp.transform(X_test)))
    print("SVM accuracy iterative: ", acc_svm)

    svm_clf.fit(knn_imp.transform(X_train), y_train)
    acc_svm = metrics.accuracy_score(y_test, svm_clf.predict(knn_imp.transform(X_test)))
    print("SVM accuracy knn: ", acc_svm)


def outliers():
    data = datasets.fetch_openml(name='diabetes', as_frame=True)
    X = data.data
    print(X)
    y = data.target
    mass = X['mass']

    zsc = zscore(X['mass'])
    list_zsc = list(zsc)
    for ind, val in enumerate(list_zsc):
        if val >= 3.0:
            list_zsc.remove(val)

    counts, bins = np.histogram(mass)
    plt.figure()
    plt.hist(bins[:-1], bins, weights=counts)
    plt.figure()
    plt.boxplot(mass)

    plt.figure()
    plt.hist(bins[:-1], bins, weights=counts)
    plt.figure()
    plt.boxplot(list_zsc)
    plt.figure()
    plt.plot(X['plas'], mass, '.b')
    plt.show()


def outliers_2():
    data = datasets.fetch_openml(name='diabetes', as_frame=True)
    X = data.data
    print(X)
    y = data.target
    mass = X['mass']
    plas = X['plas']

    x_data = np.vstack([mass, plas]).T
    print(x_data)
    forest = IsolationForest()
    forest.fit(x_data, y)

    y_predicted = []
    for data in x_data:
        y_predicted.append(forest.predict([data]))

    plt.scatter(x_data, y)
    plt.scatter(x_data, y_predicted, marker="*", c='red')

    plt.show()


if __name__ == "__main__":
    # ensamble()
    # missing_data()
    # outliers()
    outliers_2()

