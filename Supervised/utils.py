import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from sklearn import datasets


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

    return target, features