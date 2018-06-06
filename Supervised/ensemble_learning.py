import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from sklearn import linear_model, model_selection, metrics, svm, tree, ensemble, preprocessing, datasets, neural_network
from Supervised import utils


x, y = datasets.load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

mlp = ensemble.BaggingClassifier(base_estimator=neural_network.MLPClassifier(), n_estimators=10)
mlp.fit(x_train, y_train)
prediction_mlp = mlp.predict(x_test)
print(metrics.accuracy_score(y_test, prediction_mlp))

model = ensemble.BaggingClassifier(base_estimator=linear_model.LogisticRegression(), n_estimators=10)
model.fit(x_train, y_train)
prediction = model.predict(x_test)
print(metrics.accuracy_score(y_test, prediction))

vote_model = ensemble.VotingClassifier([('lg', linear_model.LogisticRegression()),
                                        ('tree', tree.DecisionTreeClassifier(max_depth=50))])
vote_model.fit(x_train, y_train)
prediction_vote = vote_model.predict(x_test)
print(metrics.accuracy_score(y_test, prediction_vote))