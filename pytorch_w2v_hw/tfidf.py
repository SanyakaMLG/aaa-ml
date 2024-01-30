import numpy as np
from collections import Counter


class TfidfVectorizer:
    def __init__(self):
        self.sorted_vocab = {}

    def fit(self, X):
        for text in X:
            for word in set(text.split()):
                if word not in self.sorted_vocab:
                    self.sorted_vocab[word] = 1
                else:
                    self.sorted_vocab[word] += 1

        self.sorted_vocab = dict(sorted(map(lambda x: (x[0], np.log(len(X) / x[1])), self.sorted_vocab.items())))
        return self

    def transform(self, X):
        res = []
        idf = self.sorted_vocab.values()
        for text in X:
            size = len(text.split())
            tf = {key: 0 for key, _ in self.sorted_vocab.items()}
            for word in text.split():
                if word in tf:
                    tf[word] += 1
            tf = list(map(lambda x: x[1] / size, tf.items()))
            res.append([a * b for a, b in zip(tf, idf)])
        return res


def read_input():
    n1, n2 = map(int, input().split())

    train_texts = [input().strip() for _ in range(n1)]
    test_texts = [input().strip() for _ in range(n2)]

    return train_texts, test_texts


def solution():
    train_texts, test_texts = read_input()
    vectorizer = TfidfVectorizer()
    vectorizer.fit(train_texts)
    transformed = vectorizer.transform(test_texts)

    for row in transformed:
        row_str = ' '.join(map(str, np.round(row, 3)))
        print(row_str)


solution()
