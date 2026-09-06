import numpy as np


class Tanh:
    def __init__(self):
        """Output this time"""

        self.output = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Tanh(x) = (e^(x) - e^(-x))/e^(x) + e^(-x)"""

        output = np.tanh(inputs)

        self.output = output
        return output

    def backward(self, incoming_gradient: np.ndarray) -> np.ndarray:
        """Tanh'(x) = (1 - (Tanh(x))^2)"""

        derivative = 1 - self.output ** 2

        out_gradient = incoming_gradient * derivative
        return out_gradient
