import numpy as np


class RNN:

    def __init__(self, in_features, hidden_size, n_classes, activation='tanh'):
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.activation = activation
        self.Wax = self.init_weight_matrix((hidden_size, in_features))
        self.Ba = self.init_weight_matrix((hidden_size, 1))
        self.Waa = self.init_weight_matrix((hidden_size, hidden_size))
        self.Way = self.init_weight_matrix((n_classes, hidden_size))
        self.By = self.init_weight_matrix((n_classes, 1))

    def init_weight_matrix(self, size):
        np.random.seed(1)
        W = np.random.uniform(size=size)
        return W

    def forward(self, x):
        res = np.zeros((self.n_classes, x.shape[1]))
        a = np.zeros((self.hidden_size, 1))
        for i in range(x.shape[1]):
            y = self.Waa @ a + self.Wax @ x[:, i].reshape(-1, 1) + self.Ba
            y = self.tanh(y)
            a = y
            y = self.Way @ y + self.By
            y = self.softmax(y)
            res[:, i] = y.reshape(-1)
        return res

    @staticmethod
    def tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    @staticmethod
    def softmax(z):
        return np.exp(z) / np.sum(np.exp(z))

def read_matrix(n_rows, dtype=float):
    return np.array([list(map(dtype, input().split())) for _ in range(n_rows)])

def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))

def solution():
    in_features, hidden_size, n_classes = map(int, input().split())
    input_vectors = read_matrix(in_features)

    rnn = RNN(in_features, hidden_size, n_classes)
    output = rnn.forward(input_vectors).round(3)
    print_matrix(output)

solution()