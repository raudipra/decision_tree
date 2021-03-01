import random

import numpy as np
import scipy.io as sio

class DigitDataset:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        mat_contents = sio.loadmat(self.file_path)
        return mat_contents['X'], mat_contents['Y']

    # Remove pixel that values always the same across all data
    def preprocess(self, X):
        feature_variance = np.var(X, axis=0)
        mask = feature_variance > 0
        X = X[:, mask]

        return X

    def fit(self, training_ratio=0.7):
        X, Y = self.load()

        X = self.preprocess(X)

        data = [(x, y) for x, y in zip(X, Y)]
        random.Random(4).shuffle(data)

        data_train = data[:int(training_ratio * len(data))]
        data_test = data[int(training_ratio * len(data)):]

        x_train, y_train, x_test, y_test = ([] for i in range(4))
        for x, y in data_train:
            x_train.append(x)
            y_train.append(y)
        x_train, y_train = np.array(x_train), np.array(y_train)

        for x, y in data_test:
            x_test.append(x)
            y_test.append(y)
        x_test, y_test = np.array(x_test), np.array(y_test)

        return x_train, y_train, x_test, y_test
