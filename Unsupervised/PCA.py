import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from sklearn import datasets, model_selection, preprocessing, decomposition, linear_model, metrics


x, y = datasets.load_wine(return_X_y=True)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3)

std = preprocessing.StandardScaler()
x_train_std = std.fit_transform(x_train)
x_test_std = std.transform(x_test)

pca = decomposition.PCA()
x_train_pca = pca.fit_transform(x_train_std)
x_test_pca = pca.transform(x_test_std)

model = linear_model.LogisticRegression()
model.fit(x_train_pca, y_train)
prediction = model.predict(x_test_pca)
print(metrics.accuracy_score(y_test, prediction))


X, y = datasets.make_moons(n_samples=100, random_state=123)

kpca = decomposition.KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_kpca = kpca.fit_transform(X)

plt.scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], color='blue', marker='o', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
