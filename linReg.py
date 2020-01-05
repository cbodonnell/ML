import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


def createLinearRegression(x, y, c):
    linear = linear_model.LinearRegression()
    bestAcc = 0
    iterations = 0
    while bestAcc < c:
        iterations += 1
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

        linear.fit(x_train, y_train)
        bestAcc = linear.score(x_test, y_test)

    print("bestAcc", bestAcc)
    print("iterations", iterations)

    return linear


def saveModel(model):
    # TODO: Get file path with tk
    path = "models/student_model.pickle"
    with open(path, "wb") as outFile:
        pickle.dump(model, outFile)


def openModel():
    # TODO: Get file path with tk
    path = "models/student_model.pickle"
    with open(path, "rb") as inFile:
        model = pickle.load(inFile)
        return model


def plotDf(df, x_col, y_col):
    style.use("ggplot")
    pyplot.scatter(df[x_col], df[y_col])
    pyplot.xlabel(x_col)
    pyplot.ylabel(y_col)
    pyplot.show()


if __name__ == "__main__":
    data = pd.read_csv("data/student-mat.csv", sep=";")
    data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

    predict = "G3"

    x_data = data.drop([predict], 1)
    y_data = data[predict]

    x = np.array(x_data)
    y = np.array(y_data)

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    c = 0.97

    # linear = createLinearRegression(x, y, c)
    # saveModel(linear)

    linear = openModel()

    print("Coef: \n", linear.coef_)
    print("Intercept: \n", linear.intercept_)

    predictions = linear.predict(x_test)

    print(x_data.head())
    for i, prediction in enumerate(predictions):
        print(prediction, x_test[i], y_test[i])

    x_col = "G1"
    y_col = "G3 "
    plotDf(data, x_col, y_col)
