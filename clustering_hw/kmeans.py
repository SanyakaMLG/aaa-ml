import numpy as np
from scipy.spatial.distance import cdist


# следует реализовать данный вид инициализации
# без нее тесты скорее всего не пройдут
def k_plus_plus(X: np.ndarray, k: int, random_state: int = 27) -> np.ndarray:
    """Инициализация центроидов алгоритмом k-means++.

    :param random_state:
    :param X: исходная выборка
    :param k: количество кластеров
    :return: набор центроидов в одном np.array
    """
    np.random.seed(random_state)
    centers = np.zeros((k, X.shape[1]))
    centers[0] = X[np.random.randint(0, X.shape[0], size=1)]
    i = 1
    while i != k:
        distances = np.array(cdist(X, centers[:i]).min(axis=1))
        p_x = np.random.random() * np.sum(distances ** 2)
        s = 0
        for idx, el in enumerate(distances ** 2):
            s += el
            if s > p_x:
                centers[i] = X[idx]
                break
        i += 1
    return centers


class KMeans:
    def __init__(self, n_clusters=8, tol=0.0001, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)

        n_samples = X.shape[0]
        n_features = X.shape[1]

        # инициализируем центры кластеров
        # centers.shape = (n_clusters, n_features)
        centers = k_plus_plus(X, self.n_clusters, self.random_state)

        for n_iter in range(self.max_iter):
            # считаем расстояние от точек из X до центроидов
            distances = cdist(X, centers)
            # определяем метки как индекс ближайшего для каждой точки центроида
            labels = distances.argmin(axis=1)

            old_centers = centers.copy()
            for c in range(self.n_clusters):
                # пересчитываем центроид
                # новый центроид есть среднее точек X с меткой рассматриваемого центроида
                centers[c, :] = np.mean(X[labels == c], axis=0)

            # записываем условие сходимости
            # норма Фробениуса разности центров кластеров двух последовательных итераций < tol
            if np.linalg.norm(old_centers - centers, ord='fro') < self.tol:
                break

        # cчитаем инерцию
        # сумма квадратов расстояний от точек до их ближайших центров кластеров
        inertia = ...

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = n_iter
        return self


    def predict(self, X):
        # определяем метку для каждого элемента X на основании обученных центров кластеров
        distances = cdist(X, self.cluster_centers_)
        labels = distances.argmin(axis=1)
        return labels

    def fit_predict(self, X):
        return self.fit(X).labels_


def read_input():
    n1, n2, k = map(int, input().split())

    read_line = lambda x: list(map(float, x.split()))
    X_train = np.array([read_line(input()) for _ in range(n1)])
    X_test = np.array([read_line(input()) for _ in range(n2)])

    return X_train, X_test, k


def solution():
    X_train, X_test, k = read_input()
    kmeans = KMeans(n_clusters=k, tol=1e-8, random_state=27)
    kmeans.fit(X_train)
    train_labels = kmeans.labels_
    test_labels = kmeans.predict(X_test)

    print(' '.join(map(str, train_labels)))
    print(' '.join(map(str, test_labels)))


solution()
