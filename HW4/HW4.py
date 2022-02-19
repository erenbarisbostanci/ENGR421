# Eren Barış Bostancı
import math
import matplotlib.pyplot as plt
import numpy as np


def RMSE(x_test, y_test, p_hat, left_borders, right_borders):
    rmse = 0
    for i in range(len(y_test)):
        test = x_test[i]
        for j in range(len(left_borders)):
            if (left_borders[j] < test) & (test <= right_borders[j]):
                rmse = rmse + (y_test[i] - p_hat[j]) * (y_test[i] - p_hat[j])
    return np.sqrt(rmse / len(y_test))


def RMSE2(x_test, y_test, p_hat, data_interval):
    rmse = 0
    for i in range(len(p_hat) - 1):
        for j in range(len(x_test)):
            test = x_test[j]
            if (data_interval[i] < test) & (test <= data_interval[i + 1]):
                rmse = rmse + (y_test[j] - p_hat[i]) * (y_test[j] - p_hat[i])

    return math.sqrt(rmse / len(y_test))


data = np.genfromtxt("hw04_data_set.csv", delimiter=",")
data = data[1:]
x_train = data[0:150, 0]
y_train = data[0:150, 1]

x_test = data[150:, 0]
y_test = data[150:, 1]

K = np.max(y_train)
N = data.shape[0]


bin_width = 0.37
origin = 1.5

minimum_value = min(x_train)
maximum_value = max(x_train)

if origin < minimum_value:
    minimum_value = origin

left_borders = np.arange(minimum_value, maximum_value, bin_width)
right_borders = np.arange(minimum_value + bin_width, maximum_value + bin_width, bin_width)


def B(x, left_border, right_border):
    tmp = np.zeros(len(x))
    for i in range(len(x)):
        if (left_border < x[i]) & (x[i] <= right_border):
            tmp[i] = 1
        else:
            tmp[i] = 0
    return tmp


p_hat_regressogram = np.zeros(len(left_borders))
for b in range(len(left_borders)):
    p_hat_regressogram[b] = np.sum(B(x_train, left_borders[b], right_borders[b]) * y_train) / np.sum(B(x_train, left_borders[b], right_borders[b]))

plt.figure(figsize=(10, 6))
plt.plot(x_train, y_train, "b.", markersize=10, label="training")
plt.plot(x_test, y_test, "r.", markersize=10, label="test")

for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [p_hat_regressogram[b], p_hat_regressogram[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [p_hat_regressogram[b], p_hat_regressogram[b + 1]], "k-")

plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption(min)")
plt.legend(loc='upper left')
plt.show()


rmse_regressogram = RMSE(x_test, y_test, p_hat_regressogram, left_borders, right_borders)

print("Regressogram => RMSE is {:f} when h is {:f}".format(rmse_regressogram, bin_width))

bin_width = 0.37

data_interval = np.linspace(minimum_value, maximum_value, 6001)


def W(x):
    for i in range(len(x)):
        if np.abs(x[i]) < 1:
            x[i] = 1
        else:
            x[i] = 0
    return x


p_hat_rms = np.zeros(len(data_interval))
for i in range(len(data_interval)):
    p_hat_rms[i] = np.sum(W(((data_interval[i] - x_train) / (bin_width / 2))) * y_train) / np.sum(W(((data_interval[i] - x_train) / (bin_width / 2))))

plt.figure(figsize=(10, 6))
plt.plot(x_train, y_train, "b.", markersize=10, label="training")
plt.plot(x_test, y_test, "r.", markersize=10, label="test")

plt.plot(data_interval, p_hat_rms, "k-")

plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption(min)")
plt.legend(loc='upper left')
plt.show()


rmse_rms = RMSE2(x_test, y_test, p_hat_rms, data_interval)

print("Running Mean Smoother => RMSE is {:f} when h is {:f}".format(rmse_rms, bin_width))


def K(x):
    for i in range(len(x)):
        x[i] = np.exp(-0.5 * (x[i] * x[i])) / np.sqrt(2 * math.pi)
    return x


p_hat_ks = np.zeros(len(data_interval))
for i in range(len(data_interval)):
    p_hat_ks[i] = np.sum(K(((data_interval[i] - x_train) / bin_width)) * y_train) / np.sum(K(((data_interval[i] - x_train) / bin_width)))


plt.figure(figsize=(10, 6))
plt.plot(x_train, y_train, "b.", markersize=10, label="training")
plt.plot(x_test, y_test, "r.", markersize=10, label="test")

plt.plot(data_interval, p_hat_ks, "k-")

plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption(min)")
plt.legend(loc='upper left')
plt.show()


rmse_ks = RMSE2(x_test, y_test, p_hat_ks, data_interval)

print("Kernel Smoother => RMSE is {:f} when h is {:f}".format(rmse_ks, bin_width))
