#Eren Barış Bostancı HW1

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(421)

# mean parameters
class_means = np.array([[+0.0, +2.5],
                        [-2.5, -2.0],
                        [+2.5, -2.0]])
# covariance parameters
class_covariances = np.array([[[+3.2, +0.0],
                               [+0.0, +1.2]],
                              [[+1.2, +0.8],
                               [+0.8, +1.2]],
                              [[+1.2, -0.8],
                               [-0.8, +1.2]]])
# sample sizes
class_sizes = np.array([120, 80, 100])


points1 = np.random.multivariate_normal(class_means[0,:], class_covariances[0,:,:], class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1,:], class_covariances[1,:,:], class_sizes[1])
points3 = np.random.multivariate_normal(class_means[2,:], class_covariances[2,:,:], class_sizes[2])
x = np.concatenate((points1, points2, points3))

# generate corresponding labels
y = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]), np.repeat(3, class_sizes[2])))

plt.figure(figsize = (10, 10))
plt.plot(points1[:,0], points1[:,1], "r.", markersize = 10)
plt.plot(points2[:,0], points2[:,1], "g.", markersize = 10)
plt.plot(points3[:,0], points3[:,1], "b.", markersize = 10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


K = 3
sample_means = []
sample_covariances = []
class_priors = []

for i in range(K):
    sample_means.append(np.mean(x[y == (i + 1)], axis=0))
    sample_covariances.append(np.cov(np.transpose(x[y == i + 1])))
    class_priors.append(np.mean(y == (i + 1)))



print("sample_means\n")
print(sample_means)
print("\n")
print("sample_covariances\n")
print(sample_covariances)
print("\n")
print("class_priors\n")
print(class_priors)

Wc = []
wc = []
wc0 = []
D = 2  # dimension

for i in range(K):
    cov_inv = np.linalg.inv(sample_covariances[i])
    Wc.append(np.array(cov_inv / -2))
    wc.append(np.matmul(cov_inv, sample_means[i]))
    wc0.append(np.array(-(np.matmul(np.matmul(np.transpose(sample_means[i]), cov_inv), sample_means[i])) * 0.5 - (D * np.log(2 * math.pi) / 2) - np.log(np.linalg.det(sample_covariances[i])) * 0.5 + np.log(class_priors[i])))

print("Wc")
print(Wc)
print("\n")
print("wc")
print(wc)
print("\n")
print("wc0")
print(wc0)
print("\n")

y_pred = []
for i in range(len(x)):
    scores = np.array([0, 0, 0])
    for j in range(K):
        score = np.matmul(np.matmul(np.transpose(x[i]), Wc[j]), x[i]) + np.matmul(np.transpose(wc[j]), x[i]) + wc0[j]
        scores[j] = score
    y_pred.append(scores)

y_pred = np.argmax(y_pred, axis=1) + 1
confusion_matrix = pd.crosstab(y_pred, y, rownames=['y_pred'], colnames=['y_truth'])
print(confusion_matrix)
