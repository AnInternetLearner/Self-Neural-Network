import numpy as np


class LeakyRelu:
    def __init__(self,alpha=0.01):
        """Alpha is how much to allow those DAMN negatives to retain their values"""

        self.input = None
        self.alpha = alpha

    def forward(self, inputs : np.ndarray) -> np.ndarray:
        """LeakyRelu(x) = max(ax,x)"""

        self.input = inputs

        output = np.maximum(self.alpha * inputs, inputs)
        return output

    def backward(self, incoming_gradient : np.ndarray) -> np.ndarray:
        """LeakyRelu'(x) = [x<=0,a,1]"""

        out_grad = incoming_gradient.copy()
        out_grad[self.input <= 0] *= self.alpha
        return out_grad
