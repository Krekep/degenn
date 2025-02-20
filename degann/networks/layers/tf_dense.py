from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import tensorflow as tf
from tensorflow import keras

from degann.config import _framework
from degann.networks import activations
from degann.networks.config_format import LAYER_DICT_NAMES


def _dec_params_to_list(
    params: Optional[Dict[str, float]]
) -> Optional[List[Tuple[str, float]]]:
    if params is None:
        return None
    res = []
    for key in params:
        res.append((key, params[key]))
    return res


def _dec_params_from_list(
    params: Optional[List[Tuple[str, float]]]
) -> Optional[Dict[str, float]]:
    if params is None:
        return None
    res = {}
    for pair in params:
        res[pair[0]] = pair[1]
    return res


class TensorflowDense(keras.layers.Layer):
    def __init__(
        self,
        input_dim=32,
        units=32,
        activation_func: str = "linear",
        weight_initializer=tf.random_normal_initializer(),
        bias_initializer=tf.random_normal_initializer(),
        is_debug=False,
        **kwargs,
    ):
        decorator_params = None

        if "decorator_params" in kwargs.keys():
            decorator_params = kwargs.get("decorator_params")
            kwargs.pop("decorator_params")

        if not isinstance(decorator_params, dict) and decorator_params is not None:
            raise TypeError(
                "Additional parameters for activation function must be dictionary"
            )

        if input_dim == 0 or units == 0:
            raise ValueError("Layer cannot have zero inputs or zero size")

        super(TensorflowDense, self).__init__(**kwargs)
        w_init = weight_initializer
        self.w = self.add_weight(
            shape=(input_dim, units),
            initializer=w_init,
            dtype="float32",
            # initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            name=f"Var_w_{self.name}",
            trainable=True,
        )
        b_init = bias_initializer
        self.b = self.add_weight(
            shape=(units,),
            initializer=b_init,
            dtype="float32",
            # initial_value=b_init(shape=(units,), dtype="float32"),
            name=f"Var_b_{self.name}",
            trainable=True,
        )

        self.units = units
        self.input_dim = input_dim
        self._is_debug = is_debug
        self.activation_func = activations.get(activation_func)
        self.activation_name = activation_func
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.decorator_params: Optional[dict] = decorator_params

    def call(self, inputs, **kwargs):
        """
        Obtaining a layer response on the input data vector
        Parameters
        ----------
        inputs
        kwargs

        Returns
        -------

        """
        if self.decorator_params is None:
            return self.activation_func(tf.matmul(inputs, self.w) + self.b)
        else:
            return self.activation_func(
                tf.matmul(inputs, self.w) + self.b, **self.decorator_params
            )

    def __str__(self):
        res = f"Layer {self.name}\n"
        res += f"weights shape = {self.w.shape}\n"
        if self._is_debug:
            # res += f"weights = {self.w.numpy()}\n"
            # res += f"biases = {self.b.numpy()}\n"
            res += f"activation = {self.activation_name}\n"
        return res

    def to_dict(self) -> dict:
        """
        Export layer to dictionary of parameters
        Returns
        -------
        config: dict
            dictionary of parameters
        """
        w = self.w.value.numpy()
        b = self.b.value.numpy()
        res = {
            LAYER_DICT_NAMES["shape"]: self.units,
            LAYER_DICT_NAMES["inp_size"]: self.input_dim,
            LAYER_DICT_NAMES["weights"]: w.tolist(),
            LAYER_DICT_NAMES["biases"]: b.tolist(),
            LAYER_DICT_NAMES["layer_type"]: type(self).__name__,
            LAYER_DICT_NAMES["dtype"]: w.dtype.name,
            LAYER_DICT_NAMES["activation"]: self.activation_name
            if self.activation_name is None
            else self.activation_name,
            LAYER_DICT_NAMES["decorator_params"]: _dec_params_to_list(
                self.decorator_params
            ),
        }

        return res

    def from_dict(self, config):
        """
        Restore layer from dictionary of parameters

        Parameters
        ----------
        config

        Returns
        -------

        """
        w = np.array(config[LAYER_DICT_NAMES["weights"]])
        b = np.array(config[LAYER_DICT_NAMES["biases"]])
        act = config[LAYER_DICT_NAMES["activation"]]
        dec_params = _dec_params_from_list(config[LAYER_DICT_NAMES["decorator_params"]])
        self.set_weights([w, b])
        # self.b = tf.Variable(
        #     initial_value=b, dtype=config[LAYER_DICT_NAMES["dtype"]], trainable=True
        # )
        self.activation_func = activations.get(act)
        self.activation_name = act
        self.decorator_params = dec_params

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def get_activation(self) -> str:
        return self.activation_name


class PyTorchDense(nn.Module):
    def __init__(
        self,
        input_dim=32,
        units=32,
        activation_func: str = "linear",
        weight_initializer=torch.nn.init.normal_,
        bias_initializer=torch.nn.init.normal_,
        is_debug=False,
        **kwargs,
    ):
        """
        Initializing the Pwtorch Dense layer.

        Parameters
        ----------
        input_dim : int
            The size of the input layer.
        units : int
            The number of neurons in the layer.
        activation_func : str
is the name of the activation function.
        weight_initializer : Callable
            The initializer of the scales.
        bias_initializer : Callable
            The offset initializer.
        is_debug : bool
            Debugging mode flag.
        kwargs : dict
            Additional parameters.
        """
        super(PyTorchDense, self).__init__()

        if input_dim == 0 or units == 0:
            raise ValueError("Layer cannot have zero inputs or zero size")

        self.units = units
        self.input_dim = input_dim
        self.is_debug = is_debug
        self.activation_func = activations.get(activation_func)
        self.activation_name = activation_func
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        # Initializing weights and offsets
        self.w = nn.Parameter(torch.empty(input_dim, units))
        self.b = nn.Parameter(torch.empty(units))

        # Using initializers
        self.weight_initializer(self.w)
        self.bias_initializer(self.b)

        self.decorator_params = kwargs.get("decorator_params", None)

    def forward(self, inputs):
        """
        The forward method for calculating the layer output.

        Parameters
        ----------
        inputs : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            The result of applying the layer.
        """
        output = torch.matmul(inputs, self.w) + self.b
        if self.decorator_params is None:
            return self.activation_func(output)
        else:
            return self.activation_func(output, **self.decorator_params)

    def __str__(self):
        res = f"Layer {self.__class__.__name__}\n"
        res += f"weights shape = {self.w.shape}\n"
        if self.is_debug:
            res += f"activation = {self.activation_name}\n"
        return res

    def to_dict(self) -> dict:
        """
        Exporting a layer to the parameter dictionary.

        Returns
        -------
        dict
            Dictionary of layer parameters.
        """
        w = self.w.detach().numpy()
        b = self.b.detach().numpy()
        res = {
            LAYER_DICT_NAMES["shape"]: self.units,
            LAYER_DICT_NAMES["inp_size"]: self.input_dim,
            LAYER_DICT_NAMES["weights"]: w.tolist(),
            LAYER_DICT_NAMES["biases"]: b.tolist(),
            LAYER_DICT_NAMES["layer_type"]: type(self).__name__,
            LAYER_DICT_NAMES["dtype"]: str(w.dtype),
            LAYER_DICT_NAMES["activation"]: self.activation_name,
            LAYER_DICT_NAMES["decorator_params"]: _dec_params_to_list(self.decorator_params),
        }
        return res

    def from_dict(self, config):
        """
        Restoring a layer from the parameter dictionary.

        Parameters
        ----------
        config : dist
            Dictionary of layer parameters.
        """
        w = torch.tensor(config[LAYER_DICT_NAMES["weights"]])
        b = torch.tensor(config[LAYER_DICT_NAMES["biases"]])
        act = config[LAYER_DICT_NAMES["activation"]]
        dec_params = _dec_params_from_list(config[LAYER_DICT_NAMES["decorator_params"]])
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(b)
        self.activation_func = activations.get(act)
        self.activation_name = act
        self.decorator_params = dec_params

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def get_activation(self) -> str:
        return self.activation_name


def create_dense_layer(**kwargs):
    """
    Creating a Dense layer depending on the framework.

    Parameters
    ----------
    kwargs : dict
        Parameters for creating a layer.

    Returns
    -------
    Layer : TensorflowDense or PyTorchDense
        A tight connection layer for the selected framework.

    Exceptions
    ----------
    Value Error
        If the framework is not supported.
    """
    if _framework == 'tensorflow':
        return TensorflowDense(**kwargs)
    elif _framework == 'pytorch':
        return PyTorchDense(**kwargs)
    else:
        raise ValueError(f"Unsupported framework: {_framework}")
