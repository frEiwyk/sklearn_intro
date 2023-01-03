import numpy as np
from sklearn.cluster import KMeans

# create a simulated dataset
np.random.seed(0)
X = np.random.normal(size=(100, 2))
X[:50] += 4
X[50:] += 1

# cluster the data using k-means
kmeans = KMeans(n_clusters=3).fit(X)

# add the cluster labels to the data
clusters = kmeans.predict(X)

# print the cluster labels
print(clusters)