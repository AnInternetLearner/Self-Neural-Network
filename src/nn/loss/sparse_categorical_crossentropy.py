import numpy as np


class SPARSE_CATEGORICAL_CROSS_ENTROPY:

    def __init__(self):
        self.size = None
        self.value_class_index = None

    def forward(self, actual: np.ndarray, prediction: np.ndarray):
        """class_index = p[arrange(y.size), y]
        loss = - mean( ln(class_index) )"""
        self.size = actual.size
        class_index = prediction[np.arange(actual.size), actual]
        self.value_class_index = class_index
        return -np.mean(np.log(class_index))

    def backward(self):
        """-1/N * class_index"""
        return -1 / (self.size * self.value_class_index)
