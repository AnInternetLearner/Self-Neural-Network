import numpy as np


class MAE:

    def __init__(self):
        """Absolute Difference and size stored"""
        self.abs_diff = None
        self.size = None

    def forward(self, prediction: np.ndarray, actual: np.ndarray):
        """|prediction - actual| * 1/n"""
        self.abs_diff = np.abs(prediction - actual)
        self.size = actual.size
        return np.mean(self.abs_diff)

    def backward(self):
        """sign(prediction - actual) * 1/n"""
        return np.sign(self.abs_diff) / self.size
