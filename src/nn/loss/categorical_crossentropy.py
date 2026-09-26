import numpy as np

class SPARSE_CROSS_ENTROPY:
    def __init__(self):
        """Need prediction and actual for backprop"""
        self.prediction = None
        self.actual = None

    def forward(self, prediction : np.ndarray, actual):
        """-mean(sum(actual * log(prediction)))"""
        self.prediction = prediction
        self.actual = actual

        return -np.mean(np.sum(actual * np.log(prediction)))

    def backward(self):
        """Well its very ez lwk Gradient = (-1 * actual/prediction) / actual.shape[0]"""
        return (- self.actual / self.prediction) / self.actual.shape[0]