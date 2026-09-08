import numpy as np
from .Initializers import Ones, Zeros, Constant, RandomNormal, RandomUniform

class DynamicInitializerError(Exception):
    """Unique error raised ONLY in 'dynamic_initializer' file, Easier Debugging. """
    pass

def error_shape_check(shape):
    if not isinstance(shape, (tuple, list)):
        raise DynamicInitializerError(f"{shape} must be a tuple/list")

    if len(shape) < 2:
        raise DynamicInitializerError(
            f"{shape} Is not a valid shape for weight Initializers, Please try use a 2D array or a greater Dimensional Tensor")

    if any(not isinstance(dim, int) for dim in shape):
        raise DynamicInitializerError("All shape dimensions must be integers")

    if any(dim <= 0 for dim in shape):
        raise DynamicInitializerError("All shape dimensions must be positive")


class Initializer:

    def __init__(self,shape: tuple):
        self.shape = shape
        error_shape_check(shape)
        self.initializer_basic = {'ones':Ones(),'zeros':Zeros(),'constant':Constant()}
        self.initializer_random = {'normal': RandomNormal, 'uniform': RandomUniform}

    def basic(self,initializer: str,
              value: float = None):

        name = initializer.lower()
        if name not in self.initializer_basic:
            raise DynamicInitializerError(f"{initializer} is not a valid initializer for Type Basic, must be one of {self.initializer_basic.keys()}")
        if name == 'constant':
            return self.initializer_basic[name](shape=self.shape, value=value)
        return self.initializer_basic[name](shape=self.shape)

    def random_initializer(self,initializer: str,
               mean: float = 0.0,
               std: float = 0.05,
               seed: int = None,
               truncate: bool = False,
               min_val: float = -1.0,
               max_val: float = 1.0,
               low : float = -0.05,
               high: float = 0.05, ):
        name = initializer.lower()
        if name not in self.initializer_random:
            raise DynamicInitializerError(f"{initializer} is not a valid initializer for Type Random, must be one of {self.initializer_random.keys()}")
        if name == 'normal':
            return self.initializer_random[name](mean=mean, std=std, seed=seed, truncate=truncate,min_val=min_val, max_val=max_val)(self.shape)
        return self.initializer_random[name](low=low,high=high,seed=seed)(self.shape)