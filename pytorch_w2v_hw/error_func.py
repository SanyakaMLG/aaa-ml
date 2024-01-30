import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)


def logloss(y, y_hat):
    return -np.sum((np.ones(y.shape) - y) * np.log(np.ones(y_hat.shape) - y_hat) + y * np.log(y_hat))


def cross_entropy(y, y_hat):
    return -np.sum(y * np.log(y_hat))


def solution():
    n, k = map(int, input().split())
    y = np.array([list(map(int, input().split())) for _ in range(n)])
    z = np.array([list(map(float, input().split())) for _ in range(n)])

    logloss_value = logloss(y, sigmoid(z))
    crossentropy_value = cross_entropy(y, softmax(z))

    logloss_value = str(np.round(logloss_value, 3))
    crossentropy_value = str(np.round(crossentropy_value, 3))
    print(logloss_value + ' ' + crossentropy_value)


solution()
