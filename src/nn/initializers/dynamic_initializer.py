import inspect
from src.nn.initializers.Initializers import Ones, Zeros, Constant, RandomNormal, RandomUniform, XavierNormal, XavierUniform, HeNormal, HeUniform, LeCunNormal, LeCunUniform

class DynamicInitializerError(Exception):
    """Unique error raised ONLY in 'dynamic_initializer' file, Easier Debugging. """
    pass

def error_shape_check(shapes):
    if not isinstance(shapes, (tuple, list)):
        raise DynamicInitializerError(f"{shapes} must be a tuple/list")

    if len(shapes) < 1:
        raise DynamicInitializerError(
            f"{shapes} Is not a valid shape for weight Initializers, Please try use a 1D array or a greater Dimensional Tensor")

    if any(not isinstance(dim, int) for dim in shapes):
        raise DynamicInitializerError("All shape dimensions must be integers")

    if any(dim <= 0 for dim in shapes):
        raise DynamicInitializerError("All shape dimensions must be positive")


class Initializer:

    def __init__(self, initializer: str,
                 value: float = None,
                 mean: float = 0.0,
                 std: float = 0.05,
                 seed: int = None,
                 truncate: bool = False,
                 min_val: float = -1.0,
                 max_val: float = 1.0,
                 low: float = -0.05,
                 high: float = 0.05,
                 gain: float = 1.0,
                 ):

        if not isinstance(initializer, str):
            raise DynamicInitializerError('Initializer Value must be a string')
        self.name = initializer.lower()
        self.value = value
        self.mean = mean
        self.std = std
        self.seed = seed
        self.truncate = truncate
        self.min_val = min_val
        self.max_val = max_val
        self.low = low
        self.high = high
        self.gain = gain

        self.initializer_basic = {'ones':Ones,'zeros':Zeros,'constant':Constant}
        self.initializer_random = {'normal': RandomNormal, 'uniform': RandomUniform}
        self.initializer_variance = {'xavier_normal': XavierNormal, 'xavier_uniform': XavierUniform, 'he_normal': HeNormal, 'he_uniform': HeUniform, 'lecun_normal': LeCunNormal, 'lecun_uniform': LeCunUniform}


    def basic(self,
              shape: tuple,):

        error_shape_check(shape)

        if self.name not in self.initializer_basic:
            raise DynamicInitializerError(f"{self.name} is not a valid initializer for Type Basic, must be one of {self.initializer_basic.keys()}")
        if 'value' in inspect.signature(
                self.initializer_basic[self.name].__call__
        ).parameters:

            if self.value is None:
                raise DynamicInitializerError("Constant initializer requires a value")

            return self.initializer_basic[self.name](value=self.value)(shape=shape)

        return self.initializer_basic[self.name]()(shape=shape)

    def random(self, shape: tuple):

        error_shape_check(shape)

        if self.name not in self.initializer_random:
            raise DynamicInitializerError(f"{self.name} is not a valid initializer for Type Random, must be one of {self.initializer_random.keys()}")
        uniform_params = {'low','high','seed'}
        if uniform_params.issubset(
                set(
                    inspect.signature(
                        self.initializer_random[self.name]
                    ).parameters
                )
        ):
            return self.initializer_random[self.name](seed=self.seed,low=self.low,high=self.high)(shape=shape)
        return self.initializer_random[self.name](seed=self.seed,min_val=self.min_val,max_val=self.max_val,truncate=self.truncate,mean=self.mean,std=self.std)(shape=shape)

    def variance(self, shape: tuple):

        error_shape_check(shape)

        if self.name not in self.initializer_variance:
            raise DynamicInitializerError(f'{self.name} is not a valid initializer for Type Variance, must be one of {self.initializer_variance.keys()}')
        return self.initializer_variance[self.name](seed=self.seed,gain=self.gain)(shape=shape)
