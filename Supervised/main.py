import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from sklearn import linear_model, model_selection, metrics, svm, tree, ensemble, preprocessing, datasets
from Supervised import utils
from mlxtend.plotting import plot_decision_regions

x, y, names = utils.dowload_iris_from_db(encode=True)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3)

model = linear_model.LogisticRegression()
model.fit(x_train, y_train)
prediction = model.predict(x_test)
score = metrics.accuracy_score(y_test, prediction)
print(score)



