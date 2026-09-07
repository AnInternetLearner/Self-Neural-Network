import numpy as np


# ____________________________
# ERROR MESSAGE
class InitializerError(Exception):
    """A very cool error message unique to Initializers and DynamicInitializer, Makes It easy to debug"""
    pass
# _____________________________


#______________________________
# Helper Functions
def calculate_fans(shape) -> tuple:
    """Calculates The fan_in, fan_out for shape of any dimension greater than 1"""
    if not len(shape) >= 2:
        raise InitializerError("XavierNormal requires shape >= 2")
    receptive_field = np.prod(shape[2:])
    fan_in = shape[1] * receptive_field
    fan_out = shape[0] * receptive_field
    return fan_in, fan_out
#______________________________


#______________________________
# Variance
class VarianceScaling:
    def __init__(self, mode: str,
                 scale: float | int = 2,
                 distribution: str = 'normal',
                 gain: float | int = 2, ):
        self.mode = mode
        self.scale = scale
        self.distribution = distribution
        self.gain = gain
        self.accepted_modes = ['fan_in', 'fan_out', 'fan_avg']
        self.accepted_distributions = ['normal', 'uniform']

    def error_checks(self, shape):

        # Scale

        if not isinstance(self.scale, (float, int)):
            raise InitializerError("Scale must be a float or int")

        if self.scale <= 0:
            raise InitializerError(
                "Scale must be positive. Because I don't want to deal with imaginary/Complex OR my values going to infinity")

        #Gain

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

        if not isinstance(shape, (tuple, list)):
            raise InitializerError(f"{shape} must be a tuple/list")

        if len(shape) < 2:
            raise InitializerError(
                f"{shape} Is not a valid shape for weight Initializers, Please try use a 2D array or a greater Dimensional Tensor")

        if any(not isinstance(dim, int) for dim in shape):
            raise InitializerError("All shape dimensions must be integers")

        if any(dim <= 0 for dim in shape):
            raise InitializerError("All shape dimensions must be positive")

    def __call__(self, shape):
        self.error_checks(shape)
        fan_in, fan_out = calculate_fans(shape)
        fan_avg = (fan_in + fan_out) / 2
        n = {'fan_in': fan_in, 'fan_out': fan_out, 'fan_avg': fan_avg}
        rng = np.random.default_rng()
        if self.distribution == 'normal':
            std = self.gain * np.sqrt(self.scale / n[self.mode])
            return rng.normal(0, std, shape)
        else:
            limit = self.gain * np.sqrt(3 * self.scale / n[self.mode])
            return rng.uniform(-limit, limit, shape)
#______________________________


# _____________________________
# Initializers
class Zeros:
    def __call__(self, shape):
        """Returns a matrix of zeros with shape {shape}"""

        return np.zeros(shape)


class Ones:
    def __call__(self, shape):
        """Returns a matrix of ones with shape {shape}"""

        return np.ones(shape)


class RandomNormal:

    def __init__(self, mean: float = 0.0, std: float = 0.05):
        self.mean = mean
        self.std = std

    def __call__(self, shape):
        """Ok so, I DON'T KNOW WHAT THE FORMULA IS ABOUT, it's late night, I am going to do this and then I will research.
        On an actual Note, this uses normal distribution to generate random numbers. And also standard deviation along with a mean to calculate
        Different values for each weight"""

        rng = np.random.default_rng()
        return rng.normal(self.mean, self.std, shape)


class RandomUniform:

    def __init__(self, low: float = -0.05, high: float = 0.05):
        self.low = low
        self.high = high

    def __call__(self, shape):
        """Random Uniform Allows Weights To be equally any amount Between Low and High
        Random Uniform = a + (b-a) * U, where a is low and b is high and U is a random float between 0 and 1"""

        rng = np.random.default_rng()
        return rng.uniform(self.low, self.high, shape)


class XavierNormal:

    def __init__(self, gain: float = 1.0):
        self.gain = gain

    def __call__(self, shape) -> np.ndarray:
        """Normal distribution where the standard deviation is equal to gain * sqrt(2/(fan_in + fan_out))
        where fan_in is the input and fan_out is the output shape"""

        value = VarianceScaling(mode='fan_avg',
                                scale=1,
                                distribution='normal',
                                gain=self.gain)
        return np.array(value(shape))

class XavierUniform:

    def __init__(self, gain: float = 1.0):
        self.gain = gain

    def __call__(self, shape) -> np.ndarray:
        """Uniform Distribution where limit is equal to gain * sqrt(6/(fan_in + fan_out)) """

        value = VarianceScaling(mode='fan_avg',
                                scale=1,
                                distribution='uniform',
                                gain=self.gain)
        return np.array(value(shape))
# ________________________________

if __name__ == '__main__':
    x = XavierUniform()
    print(x((5, 2)))
