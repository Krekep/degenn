from typing import Callable

import tensorflow as tf


def perceptron_threshold(x, threshold: float = 1.0):
    return tf.where(x >= threshold, 1.0, 0.0)


@tf.function
def parabolic(x: tf.Tensor, beta: float = 0, p: float = 1 / 5):
    """
    Activation function is described in https://rairi.frccsc.ru/en/publications/426

    Parameters
    ----------
    x: tf.Tensor
        Input data vector
    beta: float
        Offset along the OY axis
    p: float
        Focal parabola parameter

    Returns
    -------
    new_x: tf.Tensor
        Data vector after applying activation function
    """
    return tf.where(x >= 0.0, beta + tf.sqrt(2.0 * p * x), beta - tf.sqrt(-2.0 * p * x))


_activation_name = {
    "elu": tf.keras.activations.elu,
    "relu": tf.keras.activations.relu,
    "gelu": tf.keras.activations.gelu,
    "selu": tf.keras.activations.selu,
    "exponential": tf.keras.activations.exponential,
    "linear": tf.keras.activations.linear,
    "sigmoid": tf.keras.activations.sigmoid,
    "hard_sigmoid": tf.keras.activations.hard_sigmoid,
    "swish": tf.keras.activations.swish,
    "tanh": tf.keras.activations.tanh,
    "softplus": tf.keras.activations.softplus,
    "softsign": tf.keras.activations.softsign,
    "parabolic": parabolic,
    "leaky_relu": tf.nn.leaky_relu,
}


def get(name: str) -> Callable:
    """
    Get activation function by name
    Parameters
    ----------
    name: str
        name of activation function
    Returns
    -------
    func: Callable
        activation function
    """
    return _activation_name[name]


def get_all_activations() -> dict[str, Callable]:
    """
    Get all activation functions
    Parameters
    ----------

    Returns
    -------
    func: dict[str, Callable]
        dictionary of activation functions
    """
    return _activation_name
