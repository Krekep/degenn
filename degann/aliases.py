from typing import Callable, TypeAlias

import tensorflow as tf

LossFunc: TypeAlias = Callable[
    [tf.keras.Model, tf.GradientTape, tf.Tensor, tf.Tensor], float | int
]