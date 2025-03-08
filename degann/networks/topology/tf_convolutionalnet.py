import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

import os
from typing import List, Optional, Dict, Callable

from degann.networks.config_format import LAYER_DICT_NAMES
from degann.networks import layer_creator, losses, metrics, cpp_utils
from degann.networks import optimizers
from degann.networks.layers.tf_dense import TensorflowDense


class TensorflowConvolutionNet(tf.keras.Model):
    def __init__(
        self,
        output_size,
        convolution_core_size=(3, 1),
        padding_type="same",
        convolution_block_types=[],
        convolution_block_sizes=[],
        dense_block_sizes: list = [],
        chunk_size: int = 10,
        convolutional_activation_func: str = "relu",
        dense_activation_func: str = "relu",
        is_debug: bool = False,
        **kwargs,
    ):
        # input data validation
        assert len(convolution_block_types) == len(
            convolution_block_sizes
        ), "Sizes of convolutional types array and convolutional sizes array must be the same"

        # model initialisation
        super(TensorflowConvolutionNet, self).__init__()
        self.blocks = []
        self.output_size = output_size
        self.chunk_size = chunk_size
        self.trained_time = {"train_time": 0.0, "epoch_time": [], "predict_time": 0}
        print(len(convolution_block_types))
        for conv_layer_number in range(len(convolution_block_types)):
            match convolution_block_types[conv_layer_number]:
                case "maxPooling":
                    self.blocks.append(
                        layers.MaxPooling2D(
                            (convolution_block_sizes[conv_layer_number], 1)
                        )
                    )
                case "conv":
                    self.blocks.append(
                        layers.Conv2D(
                            convolution_block_sizes[conv_layer_number],
                            convolution_core_size,
                            activation=convolutional_activation_func,
                            padding=padding_type,
                        )
                    )
        self.blocks.append(layers.Flatten())
        for dense_layer_number in range(len(dense_block_sizes)):
            self.blocks.append(
                layers.Dense(
                    dense_block_sizes[dense_layer_number],
                    activation=dense_activation_func,
                ),
            )
        self.out_layer = layers.Dense(output_size, activation="linear")

    def call(self, inputs, **kwargs):
        """
        Obtaining a neural network response on the input data vector
        Parameters
        ----------
        inputs
        kwargs

        Returns
        -------

        """
        x = inputs
        for layer in self.blocks:
            x = layer(x, **kwargs)
        return self.out_layer(x, **kwargs)

    def set_name(self, new_name):
        self._name = new_name

    def split_data(self, x, y):
        x_data = np.array(
            [x[i : i + self.chunk_size] for i in range(0, len(x), self.chunk_size)]
        )[..., tf.newaxis]
        y_data = np.array(
            [y[i : i + self.chunk_size] for i in range(0, len(y), self.chunk_size)]
        )
        return [x_data, y_data]

    def fit(self, x_data, y_data, *args, **kwargs):
        x_data, y_data = self.split_data(x_data, y_data)
        super().fit(x_data, y_data, *args, **kwargs)

    def predict(self, input, *args, **kwargs):
        input, _ = self.split_data(input, np.array([]))
        return super().predict(input, *args, **kwargs)

    def __str__(self):
        res = f"IModel {self.name}\n"
        for layer in self.blocks:
            res += str(layer)
        res += str(self.out_layer)
        return res

    def custom_compile(
        self,
        rate=1e-3,
        optimizer="SGD",
        loss_func="MeanSquaredError",
        metric_funcs=None,
        run_eagerly=False,
    ):
        """
        Configures the model for training

        Parameters
        ----------
        rate: float
            learning rate for optimizer
        optimizer: str
            name of optimizer
        loss_func: str
            name of loss function
        metric_funcs: list[str]
            list with metric function names
        run_eagerly: bool

        Returns
        -------

        """
        opt = optimizers.get_optimizer(optimizer)(learning_rate=rate)
        loss = losses.get_loss(loss_func)
        m = [metrics.get_metric(metric) for metric in metric_funcs]
        self.compile(
            optimizer=opt,
            loss=loss,
            metrics=m,
            run_eagerly=run_eagerly,
        )
