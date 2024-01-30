import numpy as np


def confusion_matrix(targets, preds, t):
    new_preds = np.where(preds > t, 1, 0)
    tp = np.sum(new_preds * targets) / targets.size
    tn, fp, fn = np.zeros(targets.size), np.zeros(targets.size), np.zeros(targets.size)
    for idx, pred, target in zip(range(targets.size), new_preds, targets):
        if pred == 0 and target == 0:
            tn[idx] = 1
        elif pred == 0 and target == 1:
            fn[idx] = 1
        elif pred == 1 and target == 0:
            fp[idx] = 1
    return tp, np.sum(tn) / targets.size, np.sum(fp) / targets.size, np.sum(fn) / targets.size


def get_max_expected_value(targets, preds, ltv, c):
    mx = -np.inf
    for t in np.linspace(0, 1, 10000):
        tp, tn, fp, fn = confusion_matrix(targets, preds, t)
        max_expected_value = -c * tp + (-c) * fp + (-ltv) * fn
        if max_expected_value > mx:
            mx = max_expected_value
    return mx if mx != 0 else np.float64(0)


def read_vector(dtype=float):
    return np.array(list(map(dtype, input().split())))


def solution():
    ltv, c = map(int, input().split())
    targets = read_vector()
    predictions = read_vector()

    max_expected_value = get_max_expected_value(targets, predictions, ltv, c).round(3)
    print(max_expected_value)


solution()
