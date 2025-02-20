from typing import Callable, Dict, Optional
import torch.optim as optim
from degann.config import _framework  
from tensorflow import keras

optimizers: Dict[str, Callable] = {}

def _initialize_optimizer():
    """
    Initializes the optimizer and adds it to _optimizer_name.

    Parameters
    ----------
    name: string
        The name of the optimizer.
    """
    global optimizers

    if _framework == 'TensorFlow':
        optimizers = {
            "Adadelta": keras.optimizers.Adadelta,
            "Adafactor": keras.optimizers.Adafactor,
            "Adagrad": keras.optimizers.Adagrad,
            "Adam": keras.optimizers.Adam,
            "AdamW": keras.optimizers.AdamW,
            "Adamax": keras.optimizers.Adamax,
            "Ftrl": keras.optimizers.Ftrl,
            "Lion": keras.optimizers.Lion,
            "LossScaleOptimizer": keras.optimizers.LossScaleOptimizer,
            "Nadam": keras.optimizers.Nadam,
            "RMSprop": keras.optimizers.RMSprop,
            "SGD": keras.optimizers.SGD,
        }
    elif _framework == 'pytorch':
        optimizers = {
            "Adadelta": optim.Adadelta,
            "Adagrad": optim.Adagrad,
            "Adam": optim.Adam,
            "AdamW": optim.AdamW,
            "Adamax": optim.Adamax,
            "RMSprop": optim.RMSprop,
            "SGD": optim.SGD,
        }
    else:
        raise ValueError(f"Unsupported framework: {_framework}")

_initialize_optimizer()

def get_optimizer(name: str, **kwargs) -> Optional[Callable]:
    """
    Returns the optimizer by name, automatically initializing it if necessary.

    Parameters
    ----------
    name: string
        The name of the optimizer.
    kwargs: dict
        Parameters for initializing the optimizer.

    Returns
    -------
    optimizer: Optional[Callable]
        Initialized optimizer or None if the name is not found.
    """
    return optimizers.get(name)


def get_all_optimizers() -> Dict[str, Callable]:
    """
    Returns all available optimizers, automatically initializing them if necessary.

    Returns
    -------
    optimizers: Dict[str, Callable]
        A dictionary of all available optimizers.
    """
    return optimizers