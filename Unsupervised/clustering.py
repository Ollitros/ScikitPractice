import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import datasets, cluster, metrics
import seaborn as sns
sns.set(style='darkgrid', context='notebook')


def silhouette_plots(prediction):
    """
    Silhouette plots

    """
    cluster_labels = np.unique(prediction)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = metrics.silhouette_samples(x, prediction, metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[prediction == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(i / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper),
                 c_silhouette_vals, height=1.0, edgecolor='none', color=color)
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_silhouette_vals)
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color="red", linestyle="--")
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.show()


def elbow_method():

    """
        Elbow method

        Based on the within-cluster SSE, we can use a graphical tool, the so-called elbow
        method, to estimate the optimal number of clusters k for a given task. Intuitively,
        we can say that, if k increases, the distortion will decrease. This is because the
        samples will be closer to the centroids they are assigned to. The idea behind the
        elbow method is to identify the value of k where the distortion begins to increase
        most rapidly, which will become more clear if we plot distortion for different
        values of k

    """
    distortions = []
    for i in range(1, 11):
        model = cluster.KMeans(n_clusters=i, init='k-means++')
        model.fit(x)
        distortions.append(model.inertia_)
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow plot')
    plt.show()


def agglomerative_clustering():
    model = cluster.AgglomerativeClustering(n_clusters=4)
    prediction = model.fit_predict(x)
    plt.scatter(x[:, 0], x[:, 1], c=prediction)
    plt.title('Predicted agglomeratuve')
    plt.show()


def k_means():
    elbow_method()

    model = cluster.KMeans(n_clusters=4, init='k-means++')

    """
    init : {‘k-means++’, ‘random’ or an ndarray}
    
        Method for initialization, defaults to ‘k-means++’:
        
        ‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. See section Notes in k_init for more details.
        
        ‘random’: choose k observations (rows) at random from data for the initial centroids.
        
        If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
    
    """

    prediction = model.fit_predict(x)
    print('Distortion: %.2f' % model.inertia_)
    plt.scatter(x[:, 0], x[:, 1], c=prediction)
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='blue', marker='*', label='centroids')
    plt.legend()
    plt.title('Predicted k-means')
    plt.show()

    silhouette_plots(prediction)


def dbscan():
    x, y = datasets.make_moons(n_samples=200, noise=0.05, random_state = 0)

    db = cluster.DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
    prediction = db.fit_predict(x)

    plt.scatter(x[prediction == 0, 0], x[prediction == 0, 1], c='blue', marker='o', s=40, label='cluster 1')
    plt.scatter(x[prediction == 1, 0], x[prediction == 1, 1], c='red', marker='s', s=40, label='cluster 2')
    plt.title('Predicted dbscan')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    x, y = datasets.make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1.1)

    k_means()
    agglomerative_clustering()
    dbscan()


