from abc import ABC
from typing import Callable

import tensorflow as tf
from tensorflow import keras


@tf.function
def sign(x):
    return tf.where(x < 0.0, -1.0, 1.0)


class RelativeAbsoluteError(tf.keras.losses.Loss, ABC):
    """
    This class provides RAE loss function:
    $$ RAE = \frac{\Sum^n_{i=1} |y_i - \hat(y)_i|}{\Sum^n_{i=1} |y_i - \bar(y)|}
    """

    def __init__(self, reduction=tf.keras.losses.Reduction.NONE, name="rae", **kwargs):
        super(RelativeAbsoluteError, self).__init__(
            reduction=reduction, name=name, **kwargs
        )

    def __call__(self, y_true, y_pred, sample_weight=None):
        true_mean = tf.reduce_mean(y_true)
        squared_error_num = tf.reduce_sum(tf.abs(y_true - y_pred))
        squared_error_den = tf.reduce_sum(tf.abs(y_true - true_mean))

        squared_error_den = tf.cond(
            pred=tf.equal(squared_error_den, tf.constant(0.0)),
            true_fn=lambda: tf.constant(1.0),
            false_fn=lambda: squared_error_den,
        )

        loss = squared_error_num / squared_error_den
        return loss

    def call(self, y_true, y_pred):
        return self(y_true, y_pred)


class MaxAbsoluteDeviation(tf.keras.losses.Loss, ABC):
    """
    This class provides Max Absolute Deviation loss function:
    $$ MAD = \max |y - \hat(y)|
    """

    def __init__(
        self, reduction=tf.keras.losses.Reduction.NONE, name="my_mae", **kwargs
    ):
        super(MaxAbsoluteDeviation, self).__init__(
            reduction=reduction, name=name, **kwargs
        )

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = tf.math.reduce_max(tf.math.abs(y_true - y_pred))
        return loss

    def call(self, y_true, y_pred):
        return self(y_true, y_pred)


class MaxAbsolutePercentageError(tf.keras.losses.Loss, ABC):
    """
    This class provides Max Absolute Percentage Error loss function:
    $$ MAD = \max |\frac{y - \hat(y)}{y}|
    """

    def __init__(
        self, reduction=tf.keras.losses.Reduction.NONE, name="maxAPE", **kwargs
    ):
        super(MaxAbsolutePercentageError, self).__init__(
            reduction=reduction, name=name, **kwargs
        )

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = tf.math.reduce_max(tf.math.abs((y_true - y_pred) / y_true)) * 100.0
        return loss

    def call(self, y_true, y_pred):
        return self(y_true, y_pred)


class RMSE(tf.keras.losses.Loss, ABC):
    """
    This class provides Root Mean squared Error loss function:
    $$ MAD = \sqrt{MSE}
    """

    def __init__(self, reduction=tf.keras.losses.Reduction.NONE, name="RMSE", **kwargs):
        super(RMSE, self).__init__(reduction=reduction, name=name, **kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = tf.math.sqrt(tf.math.reduce_mean((y_pred - y_true) ** 2))
        return loss

    def call(self, y_true, y_pred):
        return self(y_true, y_pred)


# Reduction should be set to None?
_losses: dict = {
    "Huber": keras.losses.Huber(),
    "LogCosh": keras.losses.LogCosh(),
    "MeanAbsoluteError": keras.losses.MeanAbsoluteError(),
    "MeanAbsolutePercentageError": keras.losses.MeanAbsolutePercentageError(),
    "MaxAbsolutePercentageError": MaxAbsolutePercentageError(),
    "MeanSquaredError": keras.losses.MeanSquaredError(),
    "MSE": keras.losses.MeanSquaredError(),
    "RootMeanSquaredError": RMSE(),
    "RMSE": RMSE(),
    "MeanSquaredLogarithmicError": keras.losses.MeanSquaredLogarithmicError(),
    "RelativeAbsoluteError": RelativeAbsoluteError(),
    "MaxAbsoluteDeviation": MaxAbsoluteDeviation(),
    "BinaryCrossentropy": keras.losses.BinaryCrossentropy(from_logits=True),
}


def get_loss(name: str):
    """
    Get loss function by name
    Parameters
    ----------
    name: str
        Name of loss function

    Returns
    -------
    loss_class: tf.keras.losses.Loss
        Result loss function
    """
    return _losses.get(name)


def get_all_loss_functions() -> dict[str, tf.keras.losses.Loss]:
    """
    Get all loss functions
    Parameters
    ----------

    Returns
    -------
    loss_class: dict[str, tf.keras.losses.Loss]
        All loss functions
    """
    return _losses
