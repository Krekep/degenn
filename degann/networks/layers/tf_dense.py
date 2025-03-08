from typing import Optional, List, Tuple, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras

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
            LAYER_DICT_NAMES["activation"]: (
                self.activation_name
                if self.activation_name is None
                else self.activation_name
            ),
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
