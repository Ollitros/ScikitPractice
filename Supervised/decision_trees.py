import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from mpl_toolkits import mplot3d
from sklearn import linear_model, model_selection, metrics, preprocessing, tree, ensemble, datasets
from Supervised import utils


def decision_tree_classifier():

    x, y, names = utils.dowload_iris_from_db(encode=False)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3)

    model = tree.DecisionTreeClassifier(max_depth=30)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print(metrics.accuracy_score(y_test, prediction))


def random_forest_importances():

    wine = datasets.load_wine()
    x = wine.data
    y = wine.target
    feature_names = wine.feature_names
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3)

    model = ensemble.RandomForestClassifier(n_estimators=1000)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print(metrics.accuracy_score(y_test, prediction))

    importances = model.feature_importances_

    for i in range(len(importances)):
        print(feature_names[i], ' - ', (importances[i])*100, '%')

    fig = plt.figure()
    ax = plt.axes()

    plt.title('Feature Importances')
    ax.bar(range(x_train.shape[1]), importances)
    plt.xticks(range(x_train.shape[1]), feature_names, rotation=90)
    fig.show()


if __name__ == '__main__':
    random_forest_importances()