from sklearn.datasets import load_iris
from kmeans import Kmeans
import matplotlib.pyplot as plt

def main():
    data = load_iris()
    model = Kmeans(k=3)
    model.fit(data["data"])
    cluster_ids = model.cluster_ids
    sepal_height = data["data"][:, 0]
    sepal_width = data["data"][:, 1]
    plt.scatter(sepal_width, sepal_height, c=cluster_ids)
    plt.show()


if __name__ == "__main__":
    main()
    