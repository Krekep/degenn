from typing import Callable, Dict, Optional
from degann.config import _framework  
import tensorflow as tf
import torch
import torch.nn.functional as F

# An empty dictionary for activation, which will be filled on request
activations: Dict[str, Callable] = {}

def parabolic_tf(x: tf.Tensor, beta: float = 0, p: float = 1 / 5) -> tf.Tensor:
    return tf.where(x >= 0, beta + tf.sqrt(2 * p * x), beta - tf.sqrt(-2 * p * x))

def parabolic_torch(x: torch.Tensor, beta: float = 0, p: float = 1 / 5) -> torch.Tensor:
    return torch.where(x >= 0, beta + torch.sqrt(2 * p * x), beta - torch.sqrt(-2 * p * x))

# Factory for adding activations to the dictionary depending on the framework
def _initialize_activation():
    """
    Initializes the activation function and adds it to activations.
    
    Parameters
    ----------
    name: str
        Name of the activation function
    """
    global activations

    if _framework == 'TensorFlow':
        activations = {
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
            "parabolic": parabolic_tf,
        }
    elif _framework == 'PyTorch':
        activations = {
            "elu": F.elu,
            "relu": F.relu,
            "gelu": F.gelu,
            "selu": F.selu,
            "exponential": torch.exp,
            "linear": lambda x: x,
            "sigmoid": torch.sigmoid,
            "hard_sigmoid": F.hardsigmoid,
            "swish": lambda x: x * torch.sigmoid(x),
            "tanh": torch.tanh,
            "softplus": F.softplus,
            "softsign": F.softsign,
            "parabolic": parabolic_torch,
        }
    else:
        raise ValueError(f"Unsupported framework: {_framework}")

_initialize_activation()

def get(name: str) -> Optional[Callable]:
    """
    Returns the activation function by name, automatically initializing it if necessary.
        
        Parameters
        ----------
        name: string
            Name of the activation function
            
        Returns
        -------
        func: Optional[Callable]
            Activation function or None if the name is not found
    """
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations.get(name)


def get_all_activations() -> Dict[str, Callable]:
    """
    Returns all available activation functions, automatically initializing them if necessary.
        
        Returns
        -------
        func: Dict[str, Callable]
            Dictionary of all activation functions
    """
    if not activations:
        raise ValueError(f"Unsupported framework: {_framework}")
    return activations
