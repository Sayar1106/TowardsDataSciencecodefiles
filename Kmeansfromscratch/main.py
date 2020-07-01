from sklearn.datasets import load_iris, load_boston, load_diabetes
from sklearn import preprocessing
from kmeans import Kmeans
import plotly.express as px
import numpy as np

def main():
    iris = load_iris(as_frame=True)
    target = iris.target
    target = target.apply(lambda s: "Setosa" if s == 0 else ("Versicolor" if s == 1 else "Virginica"))
    scaler = preprocessing.StandardScaler().fit(iris.data.values)
    X = scaler.transform(iris.data.values)
    k = 3
    model = Kmeans(k=k)
    model.fit(X)
    cluster_ids = model.cluster_ids
    fig = px.scatter(x=X[:,0], 
                     y=X[:,1],
                     color=cluster_ids,
                     symbol=target,
                     color_continuous_scale=px.colors.sequential.Viridis,
                     opacity=0.7)
    fig.update_layout(xaxis_title="Sepal Length",
                      yaxis_title="Sepal Width",
                      coloraxis_showscale=False, 
                      title="Iris Plants (k = {})".format(k),
                      legend_title_text = "Species")
    fig.show()

    X,y = load_boston(return_X_y=True)
    X = np.delete(X, [3,8], axis=1)
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    k = 5
    model = Kmeans(k=k)
    model.fit(X)
    cluster_ids = model.cluster_ids
    cluster_ids = cluster_ids.tolist()
    cluster_ids = [str(s) for s in cluster_ids]
    fig = px.scatter(x=X[:, -1],
                     y=X[:, 3],
                     color=cluster_ids,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(xaxis_title="Median value of owner-occupied homes in $1000’s",
                      yaxis_title="nitric oxides concentration (parts per 10 million)",
                      title="Boston House Prices (k = {})".format(k),
                      legend_title_text = "Cluster ids")
    fig.show()

    X,y = load_diabetes(return_X_y=True)
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    k = 4
    model = Kmeans(k=k)
    model.fit(X)
    cluster_ids = model.cluster_ids
    cluster_ids = cluster_ids.tolist()
    cluster_ids = [str(s) for s in cluster_ids]
    fig = px.scatter(x=X[:, 0],
                     y=X[:, -1],
                     color=cluster_ids,
                     color_discrete_sequence=px.colors.qualitative.D3,
                     opacity=0.7)
    fig.update_layout(xaxis_title="Age",
                      yaxis_title="Blood sugar level",
                      title="Diabetes (k={})".format(k),
                      legend_title_text = "Cluster ids")
    fig.show()



if __name__ == "__main__":
    main()
    