import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier


def createKNNClassification(x, y, c, k_max):
    k = 0
    model = KNeighborsClassifier()
    bestAcc = 0
    iterations = 0
    while bestAcc < c:
        iterations += 1
        if k < k_max:
            k += 1
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)
        bestAcc = model.score(x_test, y_test)

    print("bestAcc", bestAcc)
    print("k", k)
    print(iterations)

    return model


if __name__ == "__main__":
    data = pd.read_csv("data/car.data")
    print(data.head())

    le = preprocessing.LabelEncoder()

    # TODO: Can refactor?
    buying = le.fit_transform(list(data["buying"]))
    maint = le.fit_transform(list(data["maint"]))
    door = le.fit_transform(list(data["door"]))
    persons = le.fit_transform(list(data["persons"]))
    lug_boot = le.fit_transform(list(data["lug_boot"]))
    safety = le.fit_transform(list(data["safety"]))
    cls = le.fit_transform(list(data["class"]))
    # print(buying)

    predict = "class"

    x = list(zip(buying, maint, door, persons, lug_boot, safety))
    y = list(cls)

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    c = 0.97
    k_max = 9

    # print(x)
    # print(y)

    model = createKNNClassification(x, y, c, k_max)

    predictions = model.predict(x_test)

    names = ["unacc", "acc", "good", "vgood"]

    for i, prediction in enumerate(predictions):
        print(names[prediction], names[y_test[i]])
        # n = model.kneighbors([x_test[i]], k_max, True)
        # print(n)
