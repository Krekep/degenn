from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

from degann.config import _framework  
import tensorflow as tf
from tensorflow import keras

import torch
import torch.nn.functional as F


# Base class for loss functions
class BaseLoss(ABC):
    """
    A base class that mimics `tf.keras.losses.Loss` but supports multiple frameworks like TensorFlow and PyTorch.
    """

    def __init__(self, reduction: str = "none", name: str = "base_loss"):
        self.reduction = reduction
        self.name = name

    @abstractmethod
    def __call__(self, y_true, y_pred):
        pass

    def reduce_loss(self, loss):
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return tf.reduce_sum(loss) if isinstance(loss, tf.Tensor) else torch.sum(loss)
        elif self.reduction == "mean":
            return tf.reduce_mean(loss) if isinstance(loss, tf.Tensor) else torch.mean(loss)
        else:
            raise ValueError(f"Unknown reduction type: {self.reduction}")

# Custom loss classes for PyTorch
class PyTorchRelativeAbsoluteError(BaseLoss):
    def __call__(self, y_true, y_pred):
        true_mean = torch.mean(y_true)
        numerator = torch.sum(torch.abs(y_true - y_pred))
        denominator = torch.sum(torch.abs(y_true - true_mean))
        denominator = denominator if denominator != 0 else torch.tensor(1.0)
        loss = numerator / denominator
        return self.reduce_loss(loss)

class PyTorchMaxAbsoluteDeviation(BaseLoss):
    def __call__(self, y_true, y_pred):
        loss = torch.max(torch.abs(y_true - y_pred))
        return self.reduce_loss(loss)

class PyTorchRMSE(BaseLoss):
    def __call__(self, y_true, y_pred):
        mse = torch.mean((y_pred - y_true) ** 2)
        loss = torch.sqrt(mse)
        return self.reduce_loss(loss)


class RelativeAbsoluteError(BaseLoss):
    def __call__(self, y_true, y_pred):
        true_mean = tf.reduce_mean(y_true)
        numerator = tf.reduce_sum(tf.abs(y_true - y_pred))
        denominator = tf.reduce_sum(tf.abs(y_true - true_mean))
        denominator = tf.where(denominator == 0.0, tf.constant(1.0), denominator)
        loss = numerator / denominator
        return self.reduce_loss(loss)

class MaxAbsoluteDeviation(BaseLoss):
    def __call__(self, y_true, y_pred):
        loss = tf.reduce_max(tf.abs(y_true - y_pred))
        return self.reduce_loss(loss)

class MaxAbsolutePercentageError(BaseLoss):
    def __call__(self, y_true, y_pred):
        loss = tf.reduce_max(tf.abs((y_true - y_pred) / y_true)) * 100.0
        return self.reduce_loss(loss)

class RMSE(BaseLoss):
    def __call__(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_pred - y_true))
        loss = tf.sqrt(mse)
        return self.reduce_loss(loss)

# Global dictionary for losses
losses: Dict[str, Callable] = {}

# Initialize losses based on framework
def _initialize_losses():
    """
    Populates the global losses dictionary based on the selected framework.
    """
    global losses

    if _framework == "TensorFlow": #enum
        losses = {
            "Huber": keras.losses.Huber(),
            "LogCosh": keras.losses.LogCosh(),
            "MeanAbsoluteError": keras.losses.MeanAbsoluteError(),
            "MeanAbsolutePercentageError": keras.losses.MeanAbsolutePercentageError(),
            "MeanSquaredError": keras.losses.MeanSquaredError(),
            "RootMeanSquaredError": RMSE(),
            "MeanSquaredLogarithmicError": keras.losses.MeanSquaredLogarithmicError(),
            "RelativeAbsoluteError": RelativeAbsoluteError(),
            "MaxAbsoluteDeviation": MaxAbsoluteDeviation(),
            "MaxAbsolutePercentageError": MaxAbsolutePercentageError(),
        }
    elif _framework == "PyTorch":
        losses = {
            "RelativeAbsoluteError": PyTorchRelativeAbsoluteError(),
            "MaxAbsoluteDeviation": PyTorchMaxAbsoluteDeviation(),
            "RootMeanSquaredError": PyTorchRMSE(),
        }
    else:
        raise ValueError(f"Unsupported framework: {_framework}")

_initialize_losses()

# Custom loss classes for TensorFlow


# Public API for retrieving losses
def get_loss(name: str) -> Optional[Callable]:
    """
    Retrieves a loss function by name.

    Parameters
    ----------
    name : str
        Name of the loss function.

    Returns
    -------
    Callable
        The corresponding loss function or None if not found.
    """
    if name not in losses:
        raise ValueError(f"Loss function '{name}' is not available in {_framework}.")
    return losses.get(name)

def get_all_loss_functions() -> Dict[str, Callable]:
    """
    Retrieves all available loss functions.

    Returns
    -------
    Dict[str, Callable]
        A dictionary containing all available loss functions.
    """
    return losses
