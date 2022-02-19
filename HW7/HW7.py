# Eren Barış Bostancı
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
import scipy.stats as stats

X = np.genfromtxt("hw07_data_set.csv", delimiter=",")

N = X.shape[0]
D = X.shape[1]
K = 5

class_means = np.array([[+2.5, +2.5],
                        [-2.5, +2.5],
                        [-2.5, -2.5],
                        [+2.5, -2.5],
                        [+0.0, +0.0]])

class_covariances = np.array([[[+0.8, -0.6],
                               [-0.6, +0.8]],
                              [[+0.8, +0.6],
                               [+0.6, +0.8]],
                              [[+0.8, -0.6],
                               [-0.6, +0.8]],
                              [[+0.8, +0.6],
                               [+0.6, +0.8]],
                              [[+1.6, +0.0],
                               [+0.0, +1.6]]])
class_sizes = np.array([50, 50, 50, 50, 100])

plt.figure(figsize=(6, 6))
plt.plot(X[:, 0], X[:, 1], "k.", markersize=10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


def update_centroids(memberships, X):
    if memberships is None:
        # initialize centroids
        centroids = np.genfromtxt("hw07_initial_centroids.csv", delimiter=",")
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

    plt.xlabel("x1")
    plt.ylabel("x2")


memberships = None
centroids = update_centroids(memberships, X)
memberships = update_memberships(centroids, X)

sample_means = centroids
sample_covariances = [None, None, None, None, None]
priors = [None, None, None, None, None]

for i in range(K):
    sample_covariances[i] = np.eye(D)
    priors[i] = class_sizes[i] / K


def E_Step(h, X, means, covariances, k):
    for n in range(N):
        h[n][k] = (stats.multivariate_normal.pdf(X[n], means[k], covariances[k]) * priors[k]) / np.sum(
            [stats.multivariate_normal.pdf(X[n], means[i], covariances[i]) * priors[i] for i in range(K)])
    return h


def M_Step(X, h, means, priors, covariances, k):
    priors[k] = np.sum(h[:, k]) / N
    means[k] = h[:, k].dot(X) / np.sum(h[:, k])
    covariances[k] = sum((h[n, k] * np.matrix((X[n] - means[k])).T.dot(np.matrix(X[n] - means[k]))) for n in range(N)) / np.sum(h[:, k])
    return means, priors, covariances


for i in range(100):
    print("Iteration: " + str(i + 1))
    h = np.zeros((N, K))
    for k in range(K):
        h = E_Step(h, X, sample_means, sample_covariances, k)
        sample_means, priors, sample_covariances = M_Step(X, h, sample_means, priors, sample_covariances, k)

memberships = update_memberships(sample_means, X)
print("sample_means:\n")
print(sample_means)

plt.figure(figsize=(6, 6))
plot_current_state(centroids, memberships, X)
x, y = np.meshgrid(np.linspace(-6, 6, 200), np.linspace(-6, 6, 200))
coordinates = np.empty(x.shape + (2,))
coordinates[:, :, 0] = x
coordinates[:, :, 1] = y
for i in range(K):
    pdf_i = stats.multivariate_normal.pdf(coordinates, class_means[i], class_covariances[i])
    pdf_f = stats.multivariate_normal.pdf(coordinates, sample_means[i], sample_covariances[i])
    plt.contour(x, y, pdf_i, linestyles='dashed', levels=[0.05])
    plt.contour(x, y, pdf_f, levels=[0.05])
plt.show()

