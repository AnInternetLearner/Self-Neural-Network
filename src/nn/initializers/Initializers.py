import numpy as np


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