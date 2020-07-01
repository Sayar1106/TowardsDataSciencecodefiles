import numpy as np
import tqdm
import itertools
import matplotlib.pyplot as plt


class Kmeans:
    def __init__(self, k=3):
        self.k = k
        self.means = None
        self._cluster_ids = None

    @property
    def cluster_ids(self):
        return self._cluster_ids

    def _init_centroid(self, m):
        return np.random.randint(0, self.k, m)

    def _cluster_means(self, X, clusters):
        m, n = X.shape[0], X.shape[1]
        # Extra column to store cluster ids
        temp = np.zeros((m, n + 1))
        temp[:, :n], temp[:, n] = X, clusters
        result = np.zeros((self.k, n))
        for i in range(self.k):
            subset = temp[np.where(temp[:, -1] == i), :n]
            if subset[0].shape[0] > 0:
                result[i] = np.mean(subset[0], axis=0)
            # Choose random data point if a cluster does not 
            # have any data associated with it
            else:
                result[i] = X[np.random.choice(X.shape[0], 1, replace=True)]

        return result

    def _compute_cluster(self, x):
        # Computes closest means to a data point x
        return min(range(self.k), key=lambda i: np.linalg.norm(x - self.means[i])**2)

    def fit(self, X, num_iterations=None):
        m = X.shape[0]
        # Initialize clusters
        initial_clusters = self._init_centroid(m)
        new_clusters = np.zeros(initial_clusters.shape)
        with tqdm.tqdm(itertools.count()) as t:
            for _ in t:
                # Compute cluster means
                self.means = self._cluster_means(X, initial_clusters)
                for i in range(m):
                    # Assign new cluster ids
                    new_clusters[i] = self._compute_cluster(X[i])
                # Check for data points that have switched cluster ids.
                count_changed = (new_clusters != initial_clusters).sum()
                if count_changed == 0:
                    break
                initial_clusters = new_clusters
                t.set_description(f"changed: {count_changed} / {X.shape[0]}")

        self._cluster_ids = new_clusters


if __name__ == "__main__":
    pass