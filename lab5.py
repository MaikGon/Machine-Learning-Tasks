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
    data = datasets.fetch_openml(name='diabetes', as_frame=True)
    X = data.data
    y = data.target
    # y[y == 'tested_positive'] = 1
    # y[y == 'tested_negative'] = 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42, stratify=y)

    # Show some info
    plt.figure()
    X_train.boxplot()
    plt.figure()
    sns.boxplot(x=X_train['mass'])
    X_train.hist()

    # Init classifiers
    forest_clf = RandomForestClassifier(random_state=42)
    forest_clf.fit(X_train, y_train)
    acc_forest = metrics.accuracy_score(y_test, forest_clf.predict(X_test))
    print("Random forest accuracy: ", acc_forest)

    tree_clf = tree.DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)
    acc_tree = metrics.accuracy_score(y_test, tree_clf.predict(X_test))
    print("Decision tree accuracy: ", acc_tree)

    svm_clf = svm.SVC(random_state=42)
    svm_clf.fit(X_train, y_train)
    acc_svm = metrics.accuracy_score(y_test, svm_clf.predict(X_test))
    print("SVM accuracy: ", acc_svm)

    # Show importances for Random Forest Classifier
    importances = forest_clf.feature_importances_
    std = np.std([trees.feature_importances_ for trees in forest_clf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f(" % (f + 1, indices[f], importances[indices[f]]))

    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices], color='r', yerr=std[indices], align='center')
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()

    # Init imputers
    simple_imp = SimpleImputer(missing_values=0.0, strategy='mean')
    simple_imp.fit(X_train[['mass']])

    iter_imp = IterativeImputer(missing_values=0.0)
    iter_imp.fit(X_train[['mass']])

    knn_imp = KNNImputer(missing_values=0.0, n_neighbors=2)
    knn_imp.fit(X_train[['mass']])

    # Test simple imputer
    X_train[['mass']] = simple_imp.transform(X_train[['mass']])

    forest_clf.fit(X_train, y_train)
    acc_forest = metrics.accuracy_score(y_test, forest_clf.predict(X_test))
    print("Random forest accuracy simple: ", acc_forest)

    tree_clf.fit(X_train, y_train)
    acc_svm = metrics.accuracy_score(y_test, tree_clf.predict(X_test))
    print("Decision tree accuracy simple: ", acc_svm)

    svm_clf.fit(X_train, y_train)
    acc_svm = metrics.accuracy_score(y_test, svm_clf.predict(X_test))
    print("SVM accuracy simple: ", acc_svm)

    # Test iterative imputer
    X_train[['mass']] = iter_imp.transform(X_train[['mass']])

    forest_clf.fit(X_train, y_train)
    acc_svm = metrics.accuracy_score(y_test, forest_clf.predict(X_test))
    print("Random forest accuracy iterative: ", acc_svm)

    tree_clf.fit(X_train, y_train)
    acc_svm = metrics.accuracy_score(y_test, tree_clf.predict(X_test))
    print("Decision tree accuracy iterative: ", acc_svm)

    svm_clf.fit(X_train, y_train)
    acc_svm = metrics.accuracy_score(y_test, svm_clf.predict(X_test))
    print("SVM accuracy iterative: ", acc_svm)

    # Test KNN imputer
    X_train[['mass']] = knn_imp.transform(X_train[['mass']])

    forest_clf.fit(X_train, y_train)
    acc_svm = metrics.accuracy_score(y_test, forest_clf.predict(X_test))
    print("Random forest accuracy knn: ", acc_svm)

    tree_clf.fit(X_train, y_train)
    acc_svm = metrics.accuracy_score(y_test, tree_clf.predict(X_test))
    print("Decision tree accuracy knn: ", acc_svm)

    svm_clf.fit(X_train, y_train)
    acc_svm = metrics.accuracy_score(y_test, svm_clf.predict(X_test))
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
    print(y)
    forest = IsolationForest(contamination='auto')
    forest.fit(x_data)

    y_predicted = []
    for data in x_data:
        y_predicted.append(forest.predict([data]))




if __name__ == "__main__":
    # ensamble()
    # missing_data()
    # outliers()
    outliers_2()

