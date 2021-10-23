import numpy as np
from scipy.sparse import *
from collections import Counter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# function to create csr matrix
def create_matrix(docs):
    n_rows = len(docs)
    index = {}
    text_id = 0
    non_zero = 0

    for d in docs:
        non_zero += len(set(d))
        for w in d:
            if w not in index:
                index[w] = text_id
                text_id += 1
    print(non_zero)

    n_cols = len(index)

    indices = np.zeros(non_zero, dtype=int)
    data = np.zeros(non_zero, dtype=int)
    indptr = np.zeros(n_rows + 1, dtype=int)

    i = 0  # document ID/row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        count = Counter(d)
        keys = list(k for k, _ in count.most_common())
        l = len(keys)
        for j, k in enumerate(keys):
            indices[j + n] = index[k]
            data[j + n] = count[k]
        indptr[i + 1] = indptr[i] + l
        n += l
        i += 1

    matrix = csr_matrix((data, indices, indptr), shape=(n_rows, n_cols), dtype=np.double)
    matrix.sort_indices()
    return matrix


def l2normalize(mat):
    mat = mat.copy()
    n_rows = mat.shape[0]
    indices, data, indptr = mat.indices, mat.data, mat.indptr

    # l2 normalization
    for i in range(n_rows):
        row_sum = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            row_sum += data[j] ** 2
        if row_sum == 0.0:
            continue  # do not normalize empty rows
        row_sum = 1.0 / np.sqrt(row_sum)
        for j in range(indptr[i], indptr[i + 1]):
            data[j] *= row_sum

    return mat


# read the dataset in key, value pair into 2 list
with open("data/train.dat", "r", encoding="utf8") as file:
    train_lines = file.readlines()

docs = list()

for row in train_lines:
    docs.append(row.rstrip().split(" "))

# call the function to convert data to csr matrix
matrix = create_matrix(docs)
matrix_l2norm = l2normalize(matrix)
matrix_dense = matrix_l2norm.toarray()

# print(matrix.shape)
# print(matrix_l2norm.shape)
# print(denseMatrix.shape)


sse_list = {}
for k in range(3, 21, 2):
    k_mean = KMeans(n_clusters=k, max_iter=21)
    k_mean.fit(matrix_dense)
    sse_list[k] = k_mean.inertia_

sse_list.items()
sse = list(sse_list.values())
ks = list(sse_list.keys())

# plotting the sse vs k graph
plt.plot(ks, sse)
plt.xlabel('Number of Clusters k')
plt.ylabel('SSE')


def find_clusters(clusters):
    clusterA = list()
    clusterB = list()
    for index in range(clusters.shape[0]):
        if clusters[index] == 0:
            clusterA.append(index)
        else:
            clusterB.append(index)

    return clusterA, clusterB


def calculate_sse(mat, clusters):
    sse_list = list()
    sse_array = []

    # calculate new root mean square error
    for cluster in clusters:
        rmse = np.sum(np.square(mat[cluster, :] - np.mean(mat[cluster, :])))
        sse_list.append(rmse)

    sse_array = np.asarray(sse_list)
    max_cluster_idx = np.argsort(sse_array)[-1]

    return max_cluster_idx


def bisecting_kmeans(matrix, k, n_iter):
    clusters = list()

    initial_cluster = list()
    for i in range(matrix.shape[0]):
        initial_cluster.append(i)

    clusters.append(initial_cluster)

    # initialize kmeans
    kmeans = KMeans(n_clusters=2, max_iter=n_iter)

    while len(clusters) < k:

        max_cluster_idx = calculate_sse(matrix, clusters)
        max_cluster = clusters[max_cluster_idx]

        kmeans.fit(matrix[max_cluster, :])
        y_kmeans = kmeans.fit_predict(matrix[max_cluster, :])
        clusters_labels = kmeans.labels_
        clusterA, clusterB = find_clusters(clusters_labels)

        clusters.pop(max_cluster_idx)

        new_clusterA = list()
        new_clusterB = list()
        for index in clusterA:
            new_clusterA.append(max_cluster[index])

        for index in clusterB:
            new_clusterB.append(max_cluster[index])

        clusters.append(new_clusterA)
        clusters.append(new_clusterB)

    labels = [0] * matrix.shape[0]

    for index, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = index + 1
    return labels


labels = bisecting_kmeans(matrix_dense, 7, 21)

# write to output file
outputFile = open("output.dat", "w")
for index in labels:
    outputFile.write(str(index) + '\n')
outputFile.close()


print("done")
