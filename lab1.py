from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn import pipeline
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def func1():
    digits = datasets.load_digits()

    # print(digits.target_names)
    # plt.imshow(digits.images[0], cmap='gray')
    # plt.show()
    clf = svm.SVC()
    clf.fit(digits.data[:-1], digits.target[:-1])
    print("Predict: ", clf.predict([digits.data[-1]])[0])
    print("Real: ", digits.target[-1])

    pickle.dump(clf, open('./clf.p', 'wb'))
    pickle.load(open('./clf.p', 'rb'))

    if clf.predict([digits.data[-1]]) == digits.target[-1]:
        print('Ok')


def lab1_4():
    olive_data = datasets.fetch_olivetti_faces()
    X_train, X_test, y_train, y_test = train_test_split(olive_data['images'], olive_data['target'],
                                                        train_size=0.8, random_state=42, shuffle=True)

    # X_train - train_images
    # X_test - test images
    # y_train - train_labes
    # y_test - test_labels

    classified_images = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [],
                         10: [], 11: [], 12: [], 13: [], 14: [], 15: [], 16: [], 17: [], 18: [], 19: [],
                         20: [], 21: [], 22: [], 23: [], 24: [], 25: [], 26: [], 27: [], 28: [], 29: [],
                         30: [], 31: [], 32: [], 33: [], 34: [], 35: [], 36: [], 37: [], 38: [], 39: [],
                         }

    for index, img in enumerate(X_test):
        classified_images[y_test[index]].append(img)

    fix, ax = plt.subplots(nrows=5, ncols=4)

    for class_id, imgs in classified_images.items():
        if class_id == 0 or class_id == 1 or class_id == 2 or class_id == 3:
            for row, image in enumerate(imgs[:5]):
                ax[row, class_id].imshow(image[..., ::-1])
                ax[row, class_id].set_title(str(class_id))
    plt.show()


def lab1_5():
    iris_toys = datasets.load_iris()

    X_train, X_test, y_train, y_test = train_test_split(iris_toys.data, iris_toys.target,
                                                        train_size=0.8, random_state=42, shuffle=True)
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    print(classifier.score(X_train, y_train))
    print(classifier.score(X_test, y_test))


def lab1_6():
    x, y = datasets.make_classification(
        n_samples=100,
        n_features=10,
        n_informative=2, n_redundant=2, n_repeated=2,
        n_classes=4,
        n_clusters_per_class=1,
        class_sep=4.0,
        flip_y=0.09
    )

    X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                        train_size=0.8, random_state=42, shuffle=True)
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    print("Train score: ", classifier.score(X_train, y_train))
    print("Test score: ", classifier.score(X_test, y_test))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=x[:, 0], ys=x[:, 1], zs=x[:, 2], c=y)
    plt.show()


def lab1_7():
    data = datasets.fetch_openml(data_id=40536, as_frame=True) # as_frame jako data frame good to know
    print(data.data) # wyswietl tabele

    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        train_size=0.8, random_state=42, shuffle=True)
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    print("Train score: ", classifier.score(X_train, y_train))
    print("Test score: ", classifier.score(X_test, y_test))


def lab1_8():
    def regression(x):
        if x >= 4.0:
            return 8.0
        else:
            return x*2

    data = np.loadtxt('trainingdata.txt', delimiter=',')

    # plt.scatter(x=data[:, 0], y=data[:, 1])
    # plt.show()

    x = data[:, 0]
    y = data[:, 1]
    y_predicted = []

    for data in x:
        y_predicted.append(regression(data))

    plt.scatter(x, y)
    plt.scatter(x, y_predicted, marker="*", c='red')
    plt.show()


def lab1_8_regresja():
    data = np.loadtxt('trainingdata.txt', delimiter=',')

    x = data[:, 0].reshape(-1, 1)
    y = data[:, 1]

    X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                     train_size=0.8, random_state=42, shuffle=True)

    # Polynomial features chyba tu by bylo dobre
    model = pipeline.Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=False))])

    model.fit(X_train, y_train)
    print("Train score Poly: ", model.score(X_train, y_train))
    print("Test score Poly: ", model.score(X_test, y_test))

    yy_predicted = []

    for data in x:
        yy_predicted.append(model.predict([data]))

    print("Mean squared error Poly: ", mean_squared_error(y, yy_predicted))
    print("Mean absolute error Poly: ", mean_absolute_error(y, yy_predicted))
    print("R2 Poly: ", r2_score(y, yy_predicted))

    classifier = SVR()
    classifier.fit(X_train, y_train)

    print("Train score SVR: ", classifier.score(X_train, y_train))
    print("Test score SVR: ", classifier.score(X_test, y_test))
    # print(classifier.predict([[3]]))

    y_predicted = []

    for data in x:
        y_predicted.append(classifier.predict([data]))

    print("Mean squared error SVR: ", mean_squared_error(y, y_predicted))
    print("Mean absolute error SVR: ", mean_absolute_error(y, y_predicted))
    print("R2 SVR: ", r2_score(y, y_predicted))

    plt.scatter(x, y)
    plt.scatter(x, y_predicted, marker="*", c='red')
    plt.scatter(x, yy_predicted,  c='yellow')
    plt.show()


if __name__ == "__main__":
    # func1()
    # lab1_4()
    # lab1_5()
    # lab1_6()
    # lab1_7()
    # lab1_8()
    lab1_8_regresja()
