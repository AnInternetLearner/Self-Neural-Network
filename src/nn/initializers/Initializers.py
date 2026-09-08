import numpy as np
from scipy.stats import truncnorm
# ____________________________
# ERROR MESSAGE
class InitializerError(Exception):
    """A very cool error message unique to Initializers and DynamicInitializer, Makes It easy to debug"""
    pass


# _____________________________


# ______________________________
# Helper Functions
def calculate_fans(shape) -> tuple:
    """Calculates The fan_in, fan_out for shape of any dimension greater than 1"""
    if not len(shape) >= 2:
        raise InitializerError("shape must be (shape) >= 2")
    receptive_field = np.prod(shape[2:])
    fan_in = shape[1] * receptive_field
    fan_out = shape[0] * receptive_field
    return fan_in, fan_out


def error_shape_check(shape):
    if not isinstance(shape, (tuple, list)):
        raise InitializerError(f"{shape} must be a tuple/list")

    if len(shape) < 2:
        raise InitializerError(
            f"{shape} Is not a valid shape for weight Initializers, Please try use a 2D array or a greater Dimensional Tensor")

    if any(not isinstance(dim, int) for dim in shape):
        raise InitializerError("All shape dimensions must be integers")

    if any(dim <= 0 for dim in shape):
        raise InitializerError("All shape dimensions must be positive")


# ______________________________


# ______________________________
# Variance
class VarianceScaling:
    def __init__(self, mode: str,
                 scale: float | int = 2,
                 distribution: str = 'normal',
                 gain: float | int = 1,
                 seed: int = None, ):
        self.mode = mode
        self.scale = scale
        self.distribution = distribution
        self.gain = gain
        self.accepted_modes = ['fan_in', 'fan_out', 'fan_avg']
        self.accepted_distributions = ['normal', 'uniform']
        self.rng = np.random.default_rng(seed)

    def error_checks(self, shape):

        # Scale

        if not isinstance(self.scale, (float, int)):
            raise InitializerError("Scale must be a float or int")

        if self.scale <= 0:
            raise InitializerError(
                "Scale must be positive. Because I don't want to deal with imaginary/Complex OR my values going to infinity")

        # Gain

        if not isinstance(self.gain, (float, int)):
            raise InitializerError("Gain must be a float or int")
        if self.gain <= 0:
            raise InitializerError('Gain Must Be Greater than 0')

        # Mode

        if not self.mode in self.accepted_modes:
            raise InitializerError(f"{self.mode} Is not an appropriate Mode, Accepted Modes are: {self.accepted_modes}")

        # Distribution

        if self.distribution not in self.accepted_distributions:
            raise InitializerError(
                f"{self.distribution} is Invalid, Accepted Distributions are: {self.accepted_distributions}")

        # Shape

        error_shape_check(shape)

    def __call__(self, shape):
        self.error_checks(shape)
        fan_in, fan_out = calculate_fans(shape)
        fan_avg = (fan_in + fan_out) / 2
        n = {'fan_in': fan_in, 'fan_out': fan_out, 'fan_avg': fan_avg}
        rng = self.rng
        if self.distribution == 'normal':
            std = self.gain * np.sqrt(self.scale / n[self.mode])
            return rng.normal(0, std, shape)
        else:
            limit = self.gain * np.sqrt(3 * self.scale / n[self.mode])
            return rng.uniform(-limit, limit, shape)


# ______________________________


# _____________________________
# Initializers


# Basic ----------------
class Zeros:
    def __call__(self, shape):
        """Returns a matrix of zeros with shape {shape}"""

        error_shape_check(shape)
        return np.zeros(shape)


class Ones:
    def __call__(self, shape):
        """Returns a matrix of ones with shape {shape}"""

        error_shape_check(shape)
        return np.ones(shape)


class Constant:
    def __call__(self, shape, value: float = 0.0):
        """Returns a matrix of {value} with shape {shape}"""
        error_shape_check(shape)
        return np.full(shape, value)


# Basic Ended ---------------


# Random ---------------------
class RandomNormal:

    def __init__(self, mean: float = 0.0,
                 std: float = 0.05,
                 seed: int = None,
                 truncate: bool = False,
                 min_val: float = -1.0,
                 max_val: float = 1.0,):
        self.mean = mean
        self.std = std
        self.rng = np.random.default_rng(seed)
        self.truncate = truncate
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, shape):
        """Ok so, I DON'T KNOW WHAT THE FORMULA IS ABOUT, it's late night, I am going to do this and then I will research.
        On an actual Note, this uses normal distribution to generate random numbers. And also standard deviation along with a mean to calculate
        Different values for each weight"""

        if self.std <= 0:
            raise InitializerError("Standard deviation must be positive")
        error_shape_check(shape)
        if not self.truncate:
            return self.rng.normal(self.mean, self.std, shape)
        if not self.min_val < self.max_val:
            raise InitializerError("Minimum value must be less than maximum value")

        a = (self.min_val - self.mean) / self.std
        b = (self.max_val - self.mean) / self.std
        return truncnorm.rvs(a, b, loc=self.mean, scale=self.std, size=shape, random_state=self.rng)


class RandomUniform:

    def __init__(self,
                 low: float = -0.05,
                 high: float = 0.05,
                 seed: int = None):
        self.low = low
        self.high = high
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape):
        """Random Uniform Allows Weights To be equally any amount Between Low and High
        Random Uniform = a + (b-a) * U, where a is low and b is high and U is a random float between 0 and 1"""

        error_shape_check(shape)
        return self.rng.uniform(self.low, self.high, shape)


# Random Ended ----------------


# Variance Based Initializer ----------
class XavierNormal:

    def __init__(self,
                 gain: float = 1.0,
                 seed: int = None):
        self.gain = gain
        self.seed = seed

    def __call__(self, shape) -> np.ndarray:
        """Normal distribution where the standard deviation is equal to gain * sqrt(2/(fan_in + fan_out))
        where fan_in is the input and fan_out is the output shape"""

        value = VarianceScaling(mode='fan_avg',
                                scale=1,
                                distribution='normal',
                                gain=self.gain,
                                seed=self.seed)
        return np.array(value(shape))


class XavierUniform:

    def __init__(self,
                 gain: float = 1.0,
                 seed: int = None):
        self.gain = gain
        self.seed = seed

    def __call__(self, shape) -> np.ndarray:
        """Uniform Distribution where limit is equal to gain * sqrt(6/(fan_in + fan_out)) """

        value = VarianceScaling(mode='fan_avg',
                                scale=1,
                                distribution='uniform',
                                gain=self.gain,
                                seed=self.seed)
        return np.array(value(shape))


class HeNormal:

    def __init__(self,
                 gain: float = 1.0,
                 seed: int = None):
        self.gain = gain
        self.seed = seed

    def __call__(self, shape) -> np.ndarray:
        """Normal distribution of variance where variance = (2/fan_in) and std is just √variance"""

        value = VarianceScaling(mode="fan_in",
                                distribution="normal",
                                scale=2,
                                gain=self.gain,
                                seed=self.seed)
        return np.array(value(shape))


class HeUniform:

    def __init__(self,
                 gain: float = 1.0,
                 seed: int = None):
        self.gain = gain
        self.seed = seed

    def __call__(self, shape) -> np.ndarray:
        """Normal distribution of variance where limit = √(6/fan_in)"""

        value = VarianceScaling(mode="fan_in",
                                distribution="uniform",
                                scale=2,
                                gain=self.gain,
                                seed=self.seed)
        return np.array(value(shape))


class LeCunNormal:

    def __init__(self,
                 gain: float = 1.0,
                 seed: int = None):
        self.gain = gain
        self.seed = seed

    def __call__(self, shape) -> np.ndarray:
        """Normal Distribution of variance where standard deviation is sqrt(1/fan_in)"""
        value = VarianceScaling(mode="fan_in",
                                distribution="normal",
                                scale=1,
                                gain=self.gain,
                                seed=self.seed
                                )

        return np.array(value(shape))


class LeCunUniform:

    def __init__(self,
                 gain: float = 1.0,
                 seed: int = None):
        self.gain = gain
        self.seed = seed

    def __call__(self, shape) -> np.ndarray:
        """Uniform Distribution where limit is sqrt(3/fan_in)"""

        value = VarianceScaling(mode="fan_in",
                                distribution="uniform",
                                scale=3,
                                gain=self.gain,
                                seed=self.seed
                                )

        return np.array(value(shape))


# Variance based initializer ended


# ________________________________
