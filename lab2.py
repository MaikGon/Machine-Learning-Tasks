from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix


def lab2_bramki():
    X = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
    y = [0, 1, 1, 1]

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    print(clf.predict([[0, 1]]))  # Sprawdź sam(a) jakie będą wyniki dla innych danych wejściowych.
    plot_tree(clf)
    plt.show()


def auta():
    # Just a simple prediction for different cars
    dict1 = {'VW': 0, 'Ford': 1, 'Opel': 2}
    dict2 = {'Wypadkowy': 1, 'Nie Wypadkowy': 0}

    # data = [['VW', 'VW', 'VW', 'Ford', 'Ford', 'Ford', 'Opel', 'Opel', 'Opel'],
    #         [10000, 1000, 10000, 100, 100000, 100000, 200000, 100, 10000],
    #         ['Wypadkowy', ' Nie Wypadkowy', 'Nie wypadkowy', 'Wypadkowy', 'Nie wypadkowy', 'Nie wypadkowy', 'Nie wypadkowy' 'Wypadkowy', 'Wypadkowy'],
    #         [0, 1, 1, 0, 1, 0, 0, 0, 0]
    #         ]

    # X = []
    # for ind, val in enumerate(data[0]):
    #     print([val, data[1][ind], data[2][ind]])
    #     X.append([val, data[1][ind], data[2][ind]])
    #
    # Y = data[3]
    # print(X)

    # This solution makes more sense
    data = [
        ['VW', 10000, 'Wypadkowy'],
        ['VW', 10000, 'Nie Wypadkowy'],
        ['Ford', 1000, 'Wypadkowy'],
        ['Ford', 100000, 'Nie Wypadkowy'],
        ['Ford', 200000, 'Nie Wypadkowy'],
        ['Opel', 100, 'Wypadkowy'],
        ['Opel', 10000, 'Nie Wypadkowy']
    ]

    Y = ['Nie kupowac', 'Kupic', 'Nie kupowac', 'Kupic', 'Nie kupowac', 'Nie kupowac', 'Kupic']

    for d in data:
        d[0] = dict1[d[0]]
        d[2] = dict2[d[2]]

    X = data

    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        train_size=0.8, random_state=42, shuffle=True)
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)

    print("Train score: ", classifier.score(X_train, y_train))
    print("Test score: ", classifier.score(X_test, y_test))
    print("Predict: ", classifier.predict([[dict1['Opel'], 100000, dict2['Wypadkowy']]]))

    plot_tree(classifier, filled=True)
    plt.show()


def digits_confusion():
    digits = datasets.load_digits()

    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                        train_size=0.7, random_state=42, shuffle=True)

    clf = svm.SVC()
    clf.fit(X_train, y_train)

    print("Predicted: ", clf.predict([digits.data[-1]])[0])
    print("True value: ", digits.target[-1])

    y_predicted = clf.predict(X_test)
    print(classification_report(y_test, y_predicted))
    print(confusion_matrix(y_test, y_predicted))

    # Show some wrong predictions
    for digit, gt, pred in zip(X_test, y_test, y_predicted):
        if gt != pred:
            print('Sample ', str(digit), 'classified as ', str(pred), 'while it should be ', str(gt))
            plt.imshow(digit.reshape(8, 8), cmap=plt.cm.gray_r)
            plt.show()

    plot_confusion_matrix(clf, X_test, y_test)
    plt.show()


if __name__ == "__main__":
    # lab2_bramki()
    # auta()
    digits_confusion()
