import numpy as np


class MSE:
    def __init__(self):
        """Stores Size and Difference of prediction - actual"""
        self.size = None
        self.diff = None

    def forward(self, prediction: np.ndarray, actual: np.ndarray):
        """1/n * sum(y_prediction - y_actual)"""
        self.size = actual.size
        self.diff = prediction - actual
        return np.mean(self.diff ** 2)

    def backward(self):
        """1/n * 2(y_prediction - y_actual)"""
        return (2 * self.diff) / self.size
