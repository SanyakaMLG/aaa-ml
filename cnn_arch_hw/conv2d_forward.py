import numpy as np


class Conv2d:

    def __init__(
            self, in_channels, out_channels, kernel_size_h, kernel_size_w, padding=0, stride=1
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.padding = padding
        self.stride = stride

        self.W, self.biases = self.init_weight_matrix()

    def init_weight_matrix(self,):
        np.random.seed(1)
        W = np.random.uniform(size=(
            self.in_channels, self.kernel_size_h, self.kernel_size_w, self.out_channels
        ))
        biases = np.random.uniform(size=(1, self.out_channels))
        return W, biases

    def forward(self, x):
        new_x = np.zeros((x.shape[0], x.shape[1] + self.padding * 2, x.shape[2] + self.padding * 2))
        new_x[:, self.padding:(new_x.shape[1] - self.padding), self.padding:(new_x.shape[2] - self.padding)] = x
        res = np.zeros(
            (self.out_channels,
             (x.shape[1] - self.kernel_size_h + 2 * self.padding + self.stride) // self.stride,
             (x.shape[2] - self.kernel_size_w + 2 * self.padding + self.stride) // self.stride)
        )
        for i in range(self.out_channels):
            for j in range(res.shape[1]):
                for k in range(res.shape[2]):
                    j_1, k_1 = j * self.stride, k * self.stride
                    res[i, j, k] = \
                        np.sum(new_x[:, j_1:j_1 + self.kernel_size_h, k_1:k_1 + self.kernel_size_w] * self.W[:, :, :, i]) + self.biases[:, i]
        return res


def read_matrix(in_channels, h, w, dtype=float):
    return np.array([list(map(dtype, input().split()))
                     for _ in range(in_channels * h)]).reshape(in_channels, h, w)


def print_matrix(matrix):
    for channel in matrix:
        for row in channel:
            print(' '.join(map(str, row)))


def solution():
    in_channels, out_channels, kernel_size_h, kernel_size_w, h, w, padding, stride = map(int, input().split())
    input_image = read_matrix(in_channels, h, w)

    conv = Conv2d(in_channels, out_channels, kernel_size_h, kernel_size_w, padding, stride)
    output = conv.forward(input_image).round(3)
    print_matrix(output)


solution()
