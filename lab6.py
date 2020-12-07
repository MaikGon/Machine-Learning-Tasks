from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn import svm, tree
import seaborn as sns
from sklearn.model_selection import train_test_split


def initial_train(data):

    data = data.drop(['boat', 'body', 'home.dest', 'name', 'ticket', 'cabin'], axis=1)

    X = data.drop(['survived'], axis=1)
    y = data['survived']
    X[X['sex'] == 'male'] = 0
    X[X['sex'] == 'female'] = 1

    X[X['embarked'] == 'male'] = 0
    X[X['embarked'] == 'female'] = 1
    X[X['embarked'] == 'male'] = 2

    print(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42, stratify=y)

    forest_clf = RandomForestClassifier(random_state=42)
    forest_clf.fit(X_train, y_train)
    acc_forest = metrics.accuracy_score(y_test, forest_clf.predict(X_test))
    print("Random forest accuracy: ", acc_forest)


def missing_values(data):
    # Show missing values
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data)

    data['embarked'].hist()
    plt.figure()
    data['age'].hist()
    # plt.show()


def replace_missing_values(data):
    # Embarked
    data['embarked'] = data['embarked'].fillna('S')

    # Cabin
    data = data.drop(['cabin'], axis=1)

    # Age


    # Fare
    value = data['fare'].astype('float64').mean()
    data['fare'] = data['fare'].fillna(value)


def label_encoder(data):
    pass


if __name__ == "__main__":
    data = pd.read_csv('titanic.csv')
    data = data.drop(['boat', 'body', 'home.dest'], axis=1)
    data = data.replace('?', np.nan)

    # initial_train(data)
    missing_values(data)
    replace_missing_values(data)
    # label_encoder(data)

