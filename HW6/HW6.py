# Eren Barış Bostancı

import cvxopt as cvx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance as dt

# Data import & variables

data = np.genfromtxt("hw06_images.csv", delimiter=",")
labels = np.genfromtxt("hw06_labels.csv", dtype=int)

X_train = data[0:1000]
y_train = labels[0:1000]

X_test = data[1000:]
y_test = labels[1000:]

N_train = len(y_train)
D_train = X_train.shape[1]

N_test = len(y_test)
D_test = X_test.shape[1]

C = 10
s = 10


def gaussian_kernel(X1, X2, s):
    D = dt.cdist(X1, X2)
    K = np.exp(-D ** 2 / (2 * s ** 2))
    return K


K_train = gaussian_kernel(X_train, X_train, s)


def lab8(y, C, s):
    yyK = np.matmul(y[:, None], y[None, :]) * K_train

    # set learning parameters
    epsilon = 1e-3

    P = cvx.matrix(yyK)
    q = cvx.matrix(-np.ones((N_train, 1)))
    G = cvx.matrix(np.vstack((-np.eye(N_train), np.eye(N_train))))
    h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C * np.ones((N_train, 1)))))
    A = cvx.matrix(1.0 * y[None, :])
    b = cvx.matrix(0.0)

    # use cvxopt library to solve QP problems
    result = cvx.solvers.qp(P, q, G, h, A, b)
    alpha = np.reshape(result["x"], N_train)
    alpha[alpha < C * epsilon] = 0
    alpha[alpha > C * (1 - epsilon)] = C

    # find bias parameter
    support_indices, = np.where(alpha != 0)
    active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
    w0 = np.mean(
        y[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))
    return alpha, w0


Y = []
for i in range(5):
    tmp = y_train.copy()
    tmp[tmp != i + 1] = -1
    tmp[tmp == i + 1] = 1
    Y.append(tmp)


F = []
Alphas = []
w0s = []
for y in Y:
    alpha, w0 = lab8(y, C, s)
    Alphas.append(alpha)
    w0s.append(w0)

    f_predicted = np.matmul(K_train, y[:, None] * alpha[:, None]) + w0
    F.append(f_predicted)
    # calculate confusion matrix
    y_predicted = 2 * (f_predicted > 0.0) - 1
    confusion_matrix = pd.crosstab(np.reshape(y_predicted, N_train), y, rownames=['y_predicted'], colnames=['y_train'])
    print(confusion_matrix)


preds_train = np.argmax(F, axis=0) + 1
confusion_matrix_train = pd.crosstab(np.reshape(preds_train, N_train), y_train, rownames=['y_predicted'], colnames=['y_train'])
print(confusion_matrix_train)


K_test = gaussian_kernel(X_test, X_train, 10)
F_test = []
f_preds = []

for i in range(len(Y)):
    y = Y[i]
    alpha = Alphas[i]
    w0 = w0s[i]
    f_predicted = np.matmul(K_test, y[:, None] * alpha[:, None]) + w0
    f_preds.append(f_predicted)


preds_test = np.argmax(f_preds, axis=0) + 1
confusion_matrix_test = pd.crosstab(np.reshape(preds_test, N_test), y_test, rownames=['y_predicted'], colnames=['y_test'])
print(confusion_matrix_test)


Cs = [0.1, 1, 10, 100, 1000]
training_cm = []
test_cm = []


for c in Cs:
    F_train = []
    Alphas = []
    w0s = []
    for y in Y:
        alpha, w0 = lab8(y, c, s)
        Alphas.append(alpha)
        w0s.append(w0)

        f_predicted = np.matmul(K_train, y[:, None] * alpha[:, None]) + w0
        F_train.append(f_predicted)
        # calculate confusion matrix
        y_predicted = 2 * (f_predicted > 0.0) - 1
        confusion_matrix = pd.crosstab(np.reshape(y_predicted, N_train), y, rownames=['y_predicted'], colnames=['y_train'])
        print(confusion_matrix)

    preds = np.argmax(F_train, axis=0)
    preds = preds + 1
    confusion_matrix = pd.crosstab(np.reshape(preds, N_train), y_train, rownames=['y_predicted'], colnames=['y_train'])
    print(confusion_matrix)
    training_cm.append(confusion_matrix)

    f_test_preds = []

    for i in range(len(Y)):
        y = Y[i]
        alpha = Alphas[i]
        w0 = w0s[i]
        f_predicted = np.matmul(K_test, y[:, None] * alpha[:, None]) + w0
        f_test_preds.append(f_predicted)
    preds_test = np.argmax(f_test_preds, axis=0) + 1
    confusion_matrix = pd.crosstab(np.reshape(preds_test, N_test), y_test, rownames=['y_predicted'], colnames=['y_test'])
    print(confusion_matrix)
    test_cm.append(confusion_matrix)

# calculate accuracy

sum = 0
accur = 0
accuracy_training = []
for cm in training_cm:
    for i in range(5):
        for j in range(5):
            sum = sum + cm.loc[i + 1, j + 1]
            if i == j:
                accur = accur + cm.loc[i + 1, j + 1]
    accuracy_training.append(accur / sum)
    accur = 0
    sum = 0

sum = 0
accur = 0
accuracy_test = []
for cm in test_cm:
    for i in range(5):
        for j in range(5):
            sum = sum + cm.loc[i + 1, j + 1]
            if i == j:
                accur = accur + cm.loc[i + 1, j + 1]
    accuracy_test.append(accur / sum)
    accur = 0
    sum = 0


# Final Prints (I printed matrixes agian since they were lost after cvxopt.solvers' prints)
print("\n")
print("Confusion Matrix Train")
print(confusion_matrix_train)
print("\n")
print("Confusion Matrix Test")
print(confusion_matrix_test)
##############################################################################################

# Plot
C_values = list(map(str, Cs))
plt.figure(figsize=(10, 6))
plt.plot(C_values, accuracy_training, "bo-", markersize=10, label="training")
plt.plot(C_values, accuracy_test, "ro-", markersize=10, label="test")
plt.xlabel("Regularization parameter (C)")
plt.ylabel("Accuracy")
plt.legend(loc='upper left')
plt.show()
