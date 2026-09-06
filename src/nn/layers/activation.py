import numpy as np

from src.nn.activations import Sigmoid, Relu, LeakyRelu, Gelu, Tanh, Swish

activations = {'sigmoid': Sigmoid,
               'relu': Relu,
               'leakyrelu': LeakyRelu,
               'tanh': Tanh,
               'swish': Swish,
               'gelu': Gelu,
               'prelu': LeakyRelu,
               'pswish': Swish
               }

# Parameters Available For CERTAIN activation functions
available_parameters = {'swish': {'beta'},
                        'leakyrelu': {'alpha'},
                        'prelu': {'alpha'},
                        'pswish': {'beta'}}


class ActivationError(Exception):
    """Unique Errors Only Resulted By the Activation Dynamic Class"""
    pass


class Activation:
    def __init__(self, activation: Relu | LeakyRelu | Gelu | Tanh | Swish | Sigmoid | str | None
                 , **kwargs):

        # ---------------
        # String
        #----------------

        if isinstance(activation, str):
            name = activation.lower()

            if name not in activations:
                raise ActivationError(f"{activation} is not a valid activation function")

            activation_class = activations[name]

            if kwargs:
                allowed = available_parameters.get(name, set())
                unexpected = set(kwargs) - allowed
                if unexpected:
                    raise ActivationError(f"Unexpected parameter {unexpected}")

                if name in ['prelu', 'pswish']:
                    self.activation = activation_class(True, **kwargs)
                else:
                    self.activation = activation_class(**kwargs)
            else:
                self.activation = activation_class()


        # ---------------
        # None
        # ----------------

        elif activation is None:
            if kwargs:
                raise ActivationError(f"Keyword argument for Activation None is Invalid")
            self.activation = None

        # ---------------
        # Activation Object
        # ----------------

        elif hasattr(activation, 'forward') and hasattr(activation, 'backward'):
            if kwargs:
                raise ActivationError(f"Keyword argument for Activation Object is invalid")
            self.activation = activation

        # ---------------
        # Invalid
        # ----------------

        else:
            raise ActivationError(f"Activation Must be a Object, String, OR a NONE value")

    def forwards(self, inputs: np.ndarray) -> np.ndarray:
        if self.activation is None:
            return inputs
        return self.activation.forward(inputs)

    def backwards(self, incoming_gradient: np.ndarray) -> np.ndarray:
        if self.activation is None:
            return incoming_gradient
        return self.activation.backward(incoming_gradient)

    @property
    def retrive_gradient(self) -> dict:
        if not hasattr(self.activation, 'gradients'):
            raise ActivationError(f"Activation must have gradient attribute")
        return self.activation.gradients

    @property
    def retrive_parameters(self) -> dict:
        if not hasattr(self.activation, 'parameters'):
            raise ActivationError(f"Activation must have parameters attribute")
        return self.activation.parameters
