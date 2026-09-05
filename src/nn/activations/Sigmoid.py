import numpy as np


class Sigmoid:
    def __init__(self):
        """Only Output Required for Derivative"""

        self.output = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Sigmoid(x) = 1/(1+e^x)"""

        output = 1 / (1 + np.exp(-inputs))

        self.output = output
        return output

    def backward(self, incoming_gradient: np.ndarray) -> np.ndarray:
        """Sigmoid'(x) = Sigmoid(x) * (1-Sigmoid(x))"""

        derivative = self.output * (1 - self.output)

        out_gradient = incoming_gradient * derivative
        return out_gradient