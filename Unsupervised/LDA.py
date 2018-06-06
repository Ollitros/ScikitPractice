import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from sklearn import datasets, model_selection, preprocessing, decomposition, linear_model, metrics, discriminant_analysis


x, y = datasets.load_wine(return_X_y=True)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3)

# Dimensional reduction
lda = discriminant_analysis.LinearDiscriminantAnalysis()
x_train_lda = lda.fit_transform(x_train, y_train)
x_test_lda = lda.transform(x_test)

# Classification model
model = linear_model.LogisticRegression()
model.fit(x_train_lda, y_train)
prediction = model.predict(x_test_lda)
score = metrics.accuracy_score(y_test, prediction)
print(score)