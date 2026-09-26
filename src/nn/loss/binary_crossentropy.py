import numpy as np


class BINARY_CROSS_ENTROPY:

    def __init__(self):
        """Backpropagation needs both inputs and size I guess"""
        self.actual = None
        self.prediction = None
        self.size = None

    def forward(self, prediction: np.ndarray, actual: np.ndarray):
        """BLE = -1 * [y ln(p) + (1-y) ln(1-p)]"""
        self.prediction = prediction
        self.actual = actual
        self.size = actual.size

        return np.mean(
            -1 * (
                    actual * np.log(prediction) + (1 - actual) * np.log(1 - prediction)
            )
        )

    def backward(self):
        """Gradient = (p - y) / p(1-p)"""

        return (
                (
                (self.prediction - self.actual)
                /
                (
                self.prediction * (1 - self.prediction)
                )
        ) / self.size)

