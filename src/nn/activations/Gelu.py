from scipy.special import erf
import numpy as np


class Gelu:
    def __init__(self):
        """A Bit more than usual"""

        self.cdf = None
        self.input = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """GELU(x) = x * Φ(x), where Φ is the standard normal CDF"""

        cdf = 0.5 * (1 + erf(inputs / np.sqrt(2)))
        output = inputs * cdf

        self.input = inputs
        self.cdf = cdf
        return output

    def backward(self, incoming_gradient: np.ndarray) -> np.ndarray:
        """GELU'(x) = cdf(x) + x*pdf(x)"""

        pdf = np.exp(-self.input ** 2 * 0.5) / np.sqrt(2 * np.pi)
        derivative = self.cdf + self.input * pdf

        out_gradient = derivative * incoming_gradient
        return out_gradient
