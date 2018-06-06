import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from mpl_toolkits import mplot3d
from sklearn import linear_model, model_selection, metrics, preprocessing
from Supervised import utils


def mnist_model():
    x, y = utils.dowload_mnist_from_db()

    # Missing data (Just for example)
    # imr = preprocessing.Imputer(missing_values='NaN')
    # imr.fit(x)
    # x = imr.transform(x)

    stand = preprocessing.StandardScaler()
    x = stand.fit_transform(x)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3)

    model = linear_model.LogisticRegression()
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    score = metrics.accuracy_score(y_test, prediction)
    print(score)

    show_errors_mnist(x_test, y_test, prediction)


def show_errors_mnist(x_test, y_test, prediction):
    condition = y_test == prediction
    indexes = np.where(condition == 0)[0]

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, n in zip(indexes, range(1, 7)):
        ax = fig.add_subplot(2, 3, n)

        image = np.reshape(x_test[i], [8, 8])
        ax.imshow(image)

        plt.title('Real: %s, Pred: %s ' % (y_test[i], prediction[i]))
    plt.show()


def iris_model():
    x, y, names = utils.dowload_iris_from_db(encode=False)

    # Encoding label data (You can don`t do this by changing encode param above)
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3)

    model = linear_model.LogisticRegression()
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    score = metrics.accuracy_score(y_test, prediction)
    print(score)

    show_errors_iris(x_test, y_test, prediction)


def show_errors_iris(x_test, y_test, prediction):
    condition = y_test == prediction
    indexes = np.where(condition == 0)[0]

    fig = plt.figure()
    ax = plt.axes()

    ax.scatter(x_test[:, 0], x_test[:, 1], cmap='viridis', c=y_test)

    for i in indexes:
        ax.scatter(x_test[i, 0], x_test[i, 1], color='red')

    fig.show()


if __name__ == '__main__':
    mnist_model()

