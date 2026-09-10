import numpy as np
from src.nn.initializers.dynamic_initializer import Initializer

class DenseError(Exception):
    """Very cool specific error to dense.py"""
    pass

class Dense:
    """OK GENERAL RULE NOW, WE FOLLOW (output, input) FOR WEIGHTS GOT IT?, ok cool."""
    def __init__(self, input_dim: int,
                 output_dim: int,
                 activation: str ='relu',
                 use_bias: bool =True,
                 weights=None,
                 bias=None ):

        initializer = Initializer
        weight_shape = (output_dim, input_dim)
        bias_shape = (output_dim,)
        if weights is None:
            self.weights = initializer('xavier_normal').initializer_variance(weight_shape)
        else:
            self.weights = weights(shape=weight_shape)
        if use_bias and bias is not None:
            self.bias = bias(shape=bias_shape)
        else:
            self.bias = initializer('zeros').initializer_basic(shape=bias_shape)
        self.activation = activation



