from numpy import maximum, ndarray


class Relu:
    def __init__(self):
        """Input needed to calculate derivative during backpropagation"""

        self.input = None

    def forward(self, inputs: ndarray) -> ndarray:
        """Relu(x) = max(0, x) or KILL ALL NEGATIVE NUMBERS"""

        self.input = inputs

        output = maximum(0, inputs)
        return output

    def backward(self, incoming_gradient: ndarray) -> ndarray:
        """Relu'(x) = [x<=0,0 | x>0,1] basically negative's have zero gradient while any positive
        has gradient of 1"""

        out_gradient = incoming_gradient.copy()

        out_gradient[self.input <= 0] = 0
        return out_gradient
