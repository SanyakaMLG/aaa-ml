import numpy as np


class Conv1d:

    def __init__(self, in_channels, out_channels, kernel_size, padding='same', activation='relu'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation

        self.W, self.biases = self.init_weight_matrix()

    def init_weight_matrix(self,):
        np.random.seed(1)
        W = np.random.uniform(size=(self.in_channels, self.kernel_size, self.out_channels))
        biases = np.random.uniform(size=(1, self.out_channels))
        return W, biases

    def forward(self, x):
        padding = self.kernel_size // 2
        out = np.zeros((self.out_channels, x.shape[1]))
        new_x = np.c_[np.zeros((self.in_channels, padding)), x, np.zeros((self.in_channels, padding))]
        for i in range(self.out_channels):
            for j in range(x.shape[1]):
                out[i, j] = np.sum(self.W[:, :, i] * new_x[:, j:j + self.kernel_size])
        out += self.biases.T
        out[out < 0] = 0
        return out


def read_matrix(n_rows, dtype=float):
    return np.array([list(map(dtype, input().split())) for _ in range(n_rows)])


def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))


def solution():
    in_channels, out_channels, kernel_size = map(int, input().split())
    input_vectors = read_matrix(in_channels)

    conv = Conv1d(in_channels, out_channels, kernel_size)
    output = conv.forward(input_vectors).round(3)
    print_matrix(output)


solution()
