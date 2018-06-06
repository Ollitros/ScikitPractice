import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from sklearn import datasets, model_selection, mixture, cluster

x, y = datasets.make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1.1)

mix = mixture.GaussianMixture(n_components=4)
prediction = mix.fit(x).predict(x)
plt.scatter(x[:, 0], x[:, 1], c=prediction)
plt.legend()
plt.title('Predicted GMM')
plt.show()

model = cluster.KMeans(n_clusters=4, init='k-means++')
prediction = model.fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=prediction)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='blue', marker='*', label='centroids')
plt.legend()
plt.title('Predicted k-means')
plt.show()