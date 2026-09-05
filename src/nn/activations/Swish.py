import numpy as np

class Swish:
    def __init__(self, beta : float = 1.0, beta_trainable : bool = False):
        """Ok so supposedly Beta is a trainable parameter or can be a constant"""

        self.inputs = None
        self.beta = beta
        self.output = None
        self.sigmoid = None
        self.beta_trainable = beta_trainable
        self.beta_gradient = None


    @property
    def parameters(self):
        return {"beta" : self.beta}


    @property
    def gradients(self):
        return {'beta' : self.beta_gradient}



    def forward(self, inputs : np.ndarray) -> np.ndarray:
        """Swish(x) = x * sigmoid(beta * x)"""

        sigmoid = 1/(1+np.exp(-inputs*self.beta))
        output = inputs * sigmoid


        self.inputs = inputs
        self.output = output
        self.sigmoid = sigmoid

        return output

    def backward(self, incoming_gradient : np.ndarray) -> np.ndarray:
        """Swish'(x) = beta * Swish(x) + sigmoid(beta * x) * (1-beta * Swish(x))"""

        derivative = self.beta * self.output + self.sigmoid * (1-self.beta * self.output)

        out_gradient = incoming_gradient * derivative

        if self.beta_trainable:
            self.beta_gradient = np.sum(incoming_gradient * self.inputs**2 * self.sigmoid * (1-self.sigmoid))

        return out_gradient

