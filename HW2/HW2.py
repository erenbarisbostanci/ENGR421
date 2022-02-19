import numpy as np
import pandas as pd
import math

data = np.genfromtxt("hw02_images.csv", delimiter=",")
labels = np.genfromtxt("hw02_labels.csv", dtype=int)

training_set = data[0:30000]
training_label = labels[0:30000]

test_set = data[30000:350000]
test_label = labels[30000:350000]


K = 5
sample_means = []
sample_deviations = []
class_priors = []

for i in range(K):
    sample_means.append(np.mean(training_set[training_label == (i + 1)], axis=0))
    sample_deviations.append(np.std(training_set[training_label == (i + 1)], axis=0))
    class_priors.append(np.mean(training_label == (i + 1)))


print("sample_means")
print(sample_means)
print("\nsample_deviations")
print(sample_deviations)
print("\nclass_priors")
print(class_priors)

training_scores = []
test_scores = []


def safelog(x):
    return (np.log(x + 1e-100))


def score(i, set):
    return np.sum(safelog((1 / np.sqrt(2 * math.pi * sample_deviations[i] * sample_deviations[i]))) -
                  (set - sample_means[i]) * (set - sample_means[i]) / (2 * sample_deviations[i] * sample_deviations[i])
                  + np.log(class_priors[i]))


def maxi(set):
    max_y = 1
    max_x = score(0, set)

    for i in range(5):
        x = score(i, set)
        if x > max_x:
            max_x = x
            max_y = i + 1
    return max_y


training_pred = []
for i in range(len(training_set)):
    training_pred.append(maxi(training_set[i]))

test_pred = []
for i in range(len(test_set)):
    test_pred.append(maxi(test_set[i]))


training_confusion_matrix = pd.crosstab(np.array(training_pred), np.array(training_label), rownames=['y_pred'], colnames=['y_truth'])
print("\ntraining_confusion_matrix: ")
print(training_confusion_matrix)


test_confusion_matrix = pd.crosstab(np.array(test_pred), np.array(test_label), rownames=['y_pred'], colnames=['y_truth'])
print("\ntest_confusion_matrix: ")
print(test_confusion_matrix)
