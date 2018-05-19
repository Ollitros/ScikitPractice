import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from sklearn import datasets
from matplotlib.colors import ListedColormap


def upload_mnist_to_db():
    features, target = datasets.load_digits(n_class=10, return_X_y=True)

    target = target.tolist()
    features = features.tolist()
    conn = psycopg2.connect(host='localhost', user='postgres', password='123', dbname='MachineLearningDB')
    cursor = conn.cursor()

    for a, b in zip(target, features):
        cursor.execute("insert into mnist(target, features) VALUES (%s, %s)", (a, b))

    conn.commit()
    conn.close()


def dowload_mnist_from_db():
    conn = psycopg2.connect(host='localhost', user='postgres', password='123', dbname='MachineLearningDB')
    cursor = conn.cursor()

    cursor.execute("select target from mnist")
    t = cursor.fetchall()
    conn.commit()
    cursor.execute("select  features from mnist")
    f = cursor.fetchall()
    conn.commit()
    conn.close()

    target = np.array(t)
    features = np.array(f)
    target = np.reshape(target, [1797])
    features = np.reshape(features, [1797, 64])

    return features, target


def upload_iris_to_db():
    data = datasets.load_iris()

    features, target = data.data, data.target
    target_names = data.target_names

    target = target.tolist()
    features = features.tolist()
    target_names = target_names.tolist()

    print(type(target_names[0]))

    print(features[0],
          features[0][0])

    conn = psycopg2.connect(host='localhost', user='postgres', password='123', dbname='MachineLearningDB')
    cursor = conn.cursor()

    for i in range(len(target)):
        cursor.execute(
            "insert into iris(sepal_length, sepal_width, petal_length, petal_width, class_name, class_number)"
            "VALUES (%s, %s, %s, %s, %s, %s)", (features[i][0], features[i][1],
                                                features[i][2], features[i][3], (target_names[target[i]]), target[i]))

    conn.commit()
    conn.close()


def dowload_iris_from_db(encode=True):

    conn = psycopg2.connect(host='localhost', user='postgres', password='123', dbname='MachineLearningDB')
    cursor = conn.cursor()

    # Getting features
    cursor.execute("select sepal_length from iris")
    sepal_length = cursor.fetchall()
    cursor.execute("select  sepal_width from iris")
    sepal_width = cursor.fetchall()
    cursor.execute("select  petal_length from iris")
    petal_length = cursor.fetchall()
    cursor.execute("select  petal_width from iris")
    petal_width = cursor.fetchall()

    # Getting targets
    cursor.execute("select class_name from iris")
    class_name = cursor.fetchall()
    cursor.execute("select class_number from iris")
    class_number = cursor.fetchall()

    conn.commit()
    conn.close()

    target = np.array(class_number)
    target_names = np.array(class_name)
    features = np.array([])
    for i in range(len(target)):
        batch = np.array([sepal_length[i], sepal_width[i],
                          petal_length[i], petal_width[i]])
        features = np.append(features, batch)

    features = np.reshape(features, [150, 4])
    target = np.reshape(target, [150])
    features_names = np.array(['sepal length', 'sepal width', 'petal length', 'petal width'])

    if encode:
        return features, target, features_names
    else:
        return features, target_names, features_names