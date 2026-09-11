import inspect
import numpy as np
from src.nn.initializers.dynamic_initializer import Initializer
from activation import Activation
class DenseError(Exception):
    """Very cool specific error to dense.py"""
    pass

def initializer_param_check(initializer_object):
    if not callable(initializer_object):
        raise TypeError(
            "Weights/Bias Initializer must be passed In convention - Initializer(params).initializer_(random/basic/variance)"
        )

    if not inspect.ismethod(initializer_object):
        raise TypeError(
            "Weights/Bias Initializer must be passed In convention - Initializer(params).initializer_(random/basic/variance)"
        )

    if not isinstance(initializer_object.__self__, Initializer):
        raise TypeError(
            "Initializer must be a method belonging to Initializer"
        )

class Dense:
    """OK GENERAL RULE NOW, WE FOLLOW (output, input) FOR WEIGHTS GOT IT?, ok cool."""
    def __init__(self, input_dim: int,
                 output_dim: int,
                 activation: str ='relu',
                 use_bias: bool =True,
                 weights=None,
                 bias=None):

        weight_shape = (output_dim, input_dim)
        bias_shape = (output_dim,)
        if weights is None:
            self.weights = Initializer(initializer='xavier_normal').variance(shape=weight_shape)
        else:
            initializer_param_check(weights)
            self.weights = weights(shape=weight_shape)
        if use_bias:
            if bias is not None:
                initializer_param_check(bias)
                self.bias = bias(shape=bias_shape)
            else:
                self.bias = Initializer(initializer='zeros').basic(shape=bias_shape)
        else:
            self.bias = None
        if activation is not None:
            self.activation = Activation(activation=activation)
        else:
            self.activation = None
        self.input = None

    def forward(self,inputs : np.ndarray) -> np.ndarray:
        self.input = inputs
        z = inputs @ self.weights.T
        if self.bias is not None:
            z += self.bias
        if self.activation is not None:
            return self.activation.forwards(z)
        return z

    def backward(self,incoming_gradient : np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.activation is not None:
            dz = self.activation.backwards(incoming_gradient)
        else:
            dz = incoming_gradient
        dw = dz.T @ self.input
        db = dz.sum(axis=0)
        dx = dz @ self.weights

        return dx, dw, db


