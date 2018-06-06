import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import sympy
from scipy import optimize
from sklearn import linear_model, model_selection, metrics, svm, tree, ensemble, preprocessing, datasets, naive_bayes
from Supervised import utils


x, y = datasets.load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

model = naive_bayes.MultinomialNB()
model.fit(x_train, y_train)
prediction = model.predict(x_test)
print(metrics.accuracy_score(y_test, prediction))