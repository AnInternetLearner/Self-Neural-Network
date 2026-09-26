from src.nn.loss import (
    SPARSE_CATEGORICAL_CROSS_ENTROPY,
    MAE,
    MSE,
    BINARY_CROSS_ENTROPY,
    CATEGORICAL_CROSS_ENTROPY)

class DynamicError(Exception):
    """A very cool special error for file dynamic_loss.py"""
    pass

class Loss:
    def __init__(self, loss_function: str = 'mse'):
        selection = 'MSE\n MAE\n BCE (binary cross entropy)\n CCE (Categorical cross entropy)\n SCCE (Sparse Categorical Cross Entropy)'
        if not isinstance(loss_function, str):
            raise DynamicError(f"loss_function must be a string from a selection of \n {selection}")
        name = loss_function.lower()
        self.dict_map = {'mse': MSE,
                         'mae': MAE,
                         'cce': CATEGORICAL_CROSS_ENTROPY,
                         'bce': BINARY_CROSS_ENTROPY,
                         'scce': SPARSE_CATEGORICAL_CROSS_ENTROPY}

