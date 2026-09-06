import numpy as np


class LeakyRelu:
    def __init__(self, alpha: float = 0.01, prelu: bool = False):
        """Alpha is how much to allow those DAMN negatives to retain their values"""

        self.alpha_trainable = prelu
        self.input = None
        self.alpha = alpha
        self.alpha_gradient = None

    @property
    def parameters(self):
        return {'alpha': self.alpha}

    @property
    def gradients(self):
        return {'alpha': self.alpha_gradient}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """LeakyRelu(x) = max(ax,x)"""

        self.input = inputs

        output = np.maximum(self.alpha * inputs, inputs)
        return output

    def backward(self, incoming_gradient: np.ndarray) -> np.ndarray:
        """LeakyRelu'(x) = [x<=0,a,1]"""

        out_grad = incoming_gradient.copy()
        out_grad[self.input <= 0] *= self.alpha

        if self.alpha_trainable:
            self.alpha_gradient = np.sum(incoming_gradient[self.input <= 0] * self.input[self.input <= 0])

        return out_grad
