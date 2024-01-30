import numpy as np


def f(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def grad_x(x, y):
    return 4 * x * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7)


def grad_y(x, y):
    return 2 * (x ** 2 + y - 11) + 4 * y * (x + y ** 2 - 7)


def find_minima(x0, y0):
    x, y = x0, y0
    v_x, v_y = 0., 0.
    tol = 1e-4
    dfdx = np.inf
    dfdy = np.inf
    iteration = 0
    lr = 0.01
    beta = 0.25

    while iteration < 200 and np.linalg.norm([dfdx, dfdy]) > tol:
        dfdx = np.clip(grad_x(x, y), -10, 10)
        dfdy = np.clip(grad_y(x, y), -10, 10)

        v_x = beta * v_x + (1 - beta) * dfdx
        v_y = beta * v_y + (1 - beta) * dfdy

        x = x - lr * v_x
        y = y - lr * v_y
        iteration += 1

    return x, y


def solution():
    x_0, y_0 = map(float, input().split())
    minima_coordinates = find_minima(x_0, y_0)
    result = ' '.join(map(str, minima_coordinates))
    print(result)


solution()