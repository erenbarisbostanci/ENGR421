# Eren Barış Bostancı

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa

X = np.genfromtxt("hw08_data_set.csv", delimiter=",")

# sample size
N = X.shape[0]
# cluster count
K = 5

plt.figure(figsize=(6, 6))
plt.plot(X[:, 0], X[:, 1], "k.", markersize=10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

S = 1.25

B = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i == j:
            B[i][j] = 0
        else:
            # euclidean distance
            distance = np.sqrt((X[j][0] - X[i][0]) ** 2 + (X[j][1] - X[i][1]) ** 2)
            if distance < S:
                B[i][j] = 1
            else:
                B[i][j] = 0

plt.figure(figsize=(6, 6))
for i in range(N):
    for j in range(N):
        if B[i][j] == 1:
            plt.plot([X[i][0], X[j][0]], [X[i][1], X[j][1]], "0.5", linewidth=0.5)
plt.plot(X[:, 0], X[:, 1], '.', markersize=10, color="black")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

D = np.zeros((N, N))
for i in range(N):
    total = 0
    for j in range(N):
        total = total + B[i][j]
    D[i][i] = total

I = np.eye(N)
L = np.subtract(D, B)
sqrt_inv_D = np.sqrt(np.linalg.inv(D))
L_sym = np.subtract(I, np.matmul(sqrt_inv_D, np.matmul(B, sqrt_inv_D)))

R = 5
eigen_vals, eigen_vecs = np.linalg.eig(L_sym)
sorted_eigen_vals = eigen_vals.argsort()
Z = eigen_vecs[:, sorted_eigen_vals[1:R + 1]]


def update_centroids(memberships, X):
    if memberships is None:
        # initialize centroids
        centroids = np.vstack([Z[28], Z[142], Z[203], Z[270], Z[276]])
    else:
        # update centroids
        centroids = np.vstack([np.mean(X[memberships == k, :], axis=0) for k in range(K)])
    return centroids


def update_memberships(centroids, X):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis=0)
    return memberships


def plot_current_state(centroids, memberships, X):
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
    if memberships is None:
        plt.plot(X[:, 0], X[:, 1], ".", markersize=10, color="black")
    else:
        for c in range(K):
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize=10,
                     color=cluster_colors[c])
    for c in range(K):
        plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize=12,
                 markerfacecolor=cluster_colors[c], markeredgecolor="black")
    plt.xlabel("x1")
    plt.ylabel("x2")


centroids = None
memberships = None
iteration = 1
while True:
    print("Iteration#{}:".format(iteration))

    old_centroids = centroids
    centroids = update_centroids(memberships, Z)
    if np.alltrue(centroids == old_centroids):
        break

    old_memberships = memberships
    memberships = update_memberships(centroids, Z)
    if np.alltrue(memberships == old_memberships):
        break

    iteration = iteration + 1

centroids = update_centroids(memberships, X)
plot_current_state(centroids, memberships, X)
plt.show()
