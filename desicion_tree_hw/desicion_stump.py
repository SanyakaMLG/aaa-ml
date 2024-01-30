import numpy as np


def H(data):
    return np.sum((data - np.mean(data)) ** 2) / data.shape[0]


def Q(left, right):
    length = len(left) + len(right)
    return H(np.hstack((left, right))) - (len(left) / length * H(left) + len(right) / length * H(right))


def choose_best(data, feature):
    best_q = 0
    best_t, preds_left, preds_right = None, None, None
    sorted_data = data[data[:, feature].argsort()]
    for i in range(1, len(sorted_data)):
        if sorted_data[i, feature] == sorted_data[i - 1, feature]:
            continue
        q = Q(sorted_data[:i, -1], sorted_data[i:, -1])
        if q >= best_q:
            best_t = (sorted_data[i - 1, feature] + sorted_data[i, feature]) / 2
            best_q = q
            preds_left = sorted_data[:i, -1].mean()
            preds_right = sorted_data[i:, -1].mean()

    return best_q, best_t, preds_left, preds_right


def decision_stump(X, y):

    arr = np.hstack((X, y))
    best_Q = 0

    for i in range(arr.shape[1] - 1):
        q, t, left, right = choose_best(arr, i)
        if q >= best_Q:
            best_Q, best_j, best_t, y_preds_left, y_preds_right = q, i, t, left, right

    best_left_ids = arr[:, best_j] < best_t
    best_right_ids = arr[:, best_j] > best_t

    result = [
        best_Q,
        best_j,
        best_t,
        best_left_ids.sum(),
        best_right_ids.sum(),
        y_preds_left,
        y_preds_right
    ]
    return result


def read_input():
    n, m = map(int, input().split())
    x_train = np.array([input().split() for _ in range(n)]).astype(float)
    y_train = np.array([input().split() for _ in range(n)]).astype(float)
    return x_train, y_train


def solution():
    for n in range(3, 20):
        for m in range(2, 5):
            for i in range(10):
                X = np.random.rand(n, m)
                y = np.random.randint(0, 2, n).reshape(-1, 1)
                result = decision_stump(X, y)
                result = np.round(result, 2)

    # result = decision_stump(X, y)
    # result = np.round(result, 2)
    # output = ' '.join(map(str, result))
    # print(output)


solution()
