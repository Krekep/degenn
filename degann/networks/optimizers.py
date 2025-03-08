from typing import Callable

from tensorflow import keras

_optimizers: dict = {
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


def get_optimizer(name: str):
    """
    Get optimizer by name
    Parameters
    ----------
    name: str
        Name of optimizer

    Returns
    -------
    optimizer_class: tf.keras.losses.Loss
        Result optimizer
    """
    if name not in _optimizers:
        raise ValueError(f"Unknown optimizer: {name}")
    return _optimizers.get(name)


def get_all_optimizers() -> dict[str, Callable]:
    """
    Get all optimizers
    Parameters
    ----------

    Returns
    -------
    optimizer_class: dict[str, tf.keras.losses.Loss]
        All optimizers
    """
    return _optimizers
