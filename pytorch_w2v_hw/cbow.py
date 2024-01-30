import numpy as np


class CBOW:

    def __init__(self, vocab_size: int, embedding_dim: int, random_state: int = 1):
        np.random.seed(random_state)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = self.init_weight_matrix()
        self.contexts = self.init_weight_matrix().T

    def init_weight_matrix(self, ):
        W = np.random.uniform(size=(self.vocab_size, self.embedding_dim))
        return W

    def forward(self, x):
        tmp = np.zeros(self.vocab_size)
        for idx in x:
            tmp[idx] += 1
        x = tmp
        x = np.sum(self.embeddings * x.reshape(-1, 1), axis=0)
        x = np.sum(self.contexts * x.reshape(-1, 1), axis=0)
        x = self.softmax(x)
        return x

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x))


def read_vector(dtype=int):
    return np.array(list(map(dtype, input().split())))


def solution():
    vocab_size, embedding_dim = read_vector()
    input_vector = read_vector()

    cbow = CBOW(vocab_size, embedding_dim)
    output = cbow.forward(input_vector).round(3)
    print(' '.join(map(str, output)))


solution()
