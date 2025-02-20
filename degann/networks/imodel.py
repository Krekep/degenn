import json
from collections import defaultdict
from typing import List, Optional, Dict, Union, Type

import numpy as np
import tensorflow as tf
from tensorflow import keras

from degann.networks.config_format import HEADER_OF_APG_FILE
from degann.networks.topology.tf_densenet import TensorflowDenseNet


def _get_act_and_init(
    kwargs: dict,
    default_act,
    default_dec: Optional[List[Optional[Dict[str, float]]]],
    default_init,
):
    if kwargs.get("activation") is None:
        activation = default_act
    else:
        activation = kwargs["activation"]
        kwargs.pop("activation")

    if kwargs.get("decorator_params") is None:
        decorator_params = default_dec
    else:
        decorator_params = kwargs["decorator_params"]
        kwargs.pop("decorator_params")

    if kwargs.get("weight") is None:
        weight = default_init
    else:
        weight = kwargs["weight"]
        kwargs.pop("weight")

    if kwargs.get("biases") is None:
        biases = default_init
    else:
        biases = kwargs["biases"]
        kwargs.pop("biases")

    return activation, decorator_params, weight, biases, kwargs


class IModel(object):
    """
    Interface class for working with neural topology
    """

    def __init__(
        self,
        input_size: int,
        block_size: List[int],
        output_size: int,
        activation_func="sigmoid",
        weight_init=tf.random_uniform_initializer(minval=-1, maxval=1),
        bias_init=tf.random_uniform_initializer(minval=-1, maxval=1),
        name="net",
        net_type="DenseNet",
        is_debug=False,
        **kwargs,
    ):
        self.network = _create_functions[net_type](
            input_size,
            block_size,
            activation_func=activation_func,
            weight=weight_init,
            biases=bias_init,
            output_size=output_size,
            is_debug=is_debug,
            **kwargs,
        )
        self._input_size = input_size
        self._output_size = output_size
        self._shape = block_size
        self._name = name
        self._is_debug = is_debug
        self.set_name(name)

    def compile(
        self,
        rate: float = 1e-2,
        optimizer: str | tf.keras.optimizers.Optimizer = "SGD",
        loss_func: str | tf.keras.losses.Loss = "MeanSquaredError",
        metrics=None,
        run_eagerly=False,
    ) -> None:
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
        metrics: list[str]
            list with metric function names
        run_eagerly: bool

        Returns
        -------

        """
        if metrics is None:
            metrics = [
                "MeanSquaredError",
                "MeanAbsoluteError",
                "MeanSquaredLogarithmicError",
            ]

        self.network.custom_compile(
            optimizer=optimizer,
            rate=rate,
            loss_func=loss_func,
            metric_funcs=metrics,
            run_eagerly=run_eagerly,
        )

    def feedforward(self, inputs: np.ndarray) -> tf.Tensor:
        """
        Return network answer for passed input by network __call__()

        Parameters
        ----------
        inputs: np.ndarray
            Input activation vector

        Returns
        -------
        outputs: tf.Tensor
            Network answer
        """

        return self.network(inputs, training=False)

    def predict(
        self, inputs: np.ndarray, callbacks: Optional[List] = None
    ) -> np.ndarray:
        """
        Return network answer for passed input by network predict()

        Parameters
        ----------
        inputs: np.ndarray
            Input activation vector
        callbacks: list
            List of tensorflow callbacks for predict function

        Returns
        -------
        outputs: np.ndarray
            Network answer
        """

        return self.network.predict(inputs, verbose=0, callbacks=callbacks)

    def train(
        self,
        x_data: np.ndarray | tf.Tensor,
        y_data: np.ndarray | tf.Tensor,
        validation_split=0.0,
        validation_data=None,
        epochs=10,
        mini_batch_size=None,
        callbacks: Optional[List] = None,
        verbose="auto",
    ) -> keras.callbacks.History:
        """
        Train network on passed dataset and return training history

        Parameters
        ----------
        x_data: np.ndarray
            Array of input vectors
        y_data: np.ndarray
            Array of output vectors
        validation_split: float
            Percentage of data to validate
        validation_data: tuple[np.ndarray, np.ndarray]
            Validation dataset
        epochs: int
            Count of epochs for training
        mini_batch_size: int
            Size of batches
        callbacks: list
            List of tensorflow callbacks for fit function
        verbose: int
            Output accompanying training

        Returns
        -------
        history: tf.keras.callbacks.History
            History of training
        """
        if self._is_debug:
            if callbacks is not None:
                callbacks.append(
                    tf.keras.callbacks.CSVLogger(
                        f"log_{self.get_name}.csv", separator=",", append=False
                    )
                )
            else:
                callbacks = [
                    tf.keras.callbacks.CSVLogger(
                        f"log_{self.get_name}.csv", separator=",", append=False
                    )
                ]
        temp = self.network.fit(
            x_data,
            y_data,
            batch_size=mini_batch_size,
            callbacks=callbacks,
            validation_split=validation_split,
            validation_data=validation_data,
            epochs=epochs,
            verbose=verbose,
        )
        return temp

    def evaluate(
        self,
        x_data: np.ndarray | tf.Tensor,
        y_data: np.ndarray | tf.Tensor,
        batch_size=None,
        callbacks: Optional[List] = None,
        verbose="auto",
        **kwargs,
    ) -> dict[str, float]:
        """
        Evaluate network on passed dataset and return evaluate history

        Parameters
        ----------
        x_data: np.ndarray
            Array of input vectors
        y_data: np.ndarray
            Array of output vectors
        batch_size: int
            Size of batches
        callbacks: list
            List of tensorflow callbacks for evaluate function
        verbose: int
            Output accompanying evaluating

        Returns
        -------
        history: dict[str, float]
            Scalar validation loss
        """
        if self._is_debug:
            if callbacks is not None:
                callbacks.append(
                    tf.keras.callbacks.CSVLogger(
                        f"log_{self.get_name}.csv", separator=",", append=False
                    )
                )
            else:
                callbacks = [
                    tf.keras.callbacks.CSVLogger(
                        f"log_{self.get_name}.csv", separator=",", append=False
                    )
                ]
        # In debug evaluate returns the dictionary of metric (and loss) values on validation data
        self._evaluate_history: dict[str, float] = self.network.evaluate(  # type: ignore
            x_data,
            y_data,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            **kwargs,
        )
        return self._evaluate_history

    def clear_history(self):
        del self.network.history
        del self._evaluate_history

    def export_to_cpp(
        self,
        path: str,
        array_type: str = "[]",
        path_to_compiler: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Export neural network as feedforward function on c++

        Parameters
        ----------
        path: str
            path to file with name, without extension
        array_type: str
            c-style or cpp-style ("[]" or "vector")
        path_to_compiler: str
            path to c/c++ compiler, if `None` then the resulting code will not be compiled
        kwargs

        Returns
        -------

        """
        self.network.export_to_cpp(path, array_type, path_to_compiler)

    def to_dict(self, **kwargs):
        """
        Export neural network to dictionary

        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        return self.network.to_dict(**kwargs)

    def export_to_file(self, path, **kwargs):
        """
        Export neural network as parameters to file

        Parameters
        ----------
        path:
            path to file with name, without extension
        kwargs

        Returns
        -------

        """
        config = self.to_dict(**kwargs)
        with open(path + ".apg", "w") as f:
            f.write(HEADER_OF_APG_FILE + json.dumps(config, indent=2))

    def from_dict(self, config: dict, **kwargs):
        """
        Import neural network from dictionary

        Parameters
        ----------
        config: dict
            Network configuration

        Returns
        -------

        """
        self._shape = config["block_size"]
        self.network.from_dict(config, **kwargs)

    def from_file(self, path: str, **kwargs):
        """
        Import neural network as parameters from file

        Parameters
        ----------
        path:
            path to file with name, without extension
        kwargs

        Returns
        -------

        """
        with open(path + ".apg", "r") as f:
            for header in range(HEADER_OF_APG_FILE.count("\n")):
                _ = f.readline()
            config_str = ""
            for line in f:
                config_str += line
            config = json.loads(config_str)
            self.network.from_dict(config)
            self.set_name(config["name"])

    def set_name(self, name: str) -> None:
        """
        Set network name

        Parameters
        ----------
        name: str
            New name
        Returns
        -------
        None
        """
        self.network.set_name(name)
        self._name = name

    @property
    def get_name(self) -> str:
        return self._name

    @property
    def get_shape(self) -> List[int]:
        """
        Get shape for current network

        Returns
        -------
        shape: List[int]
            Network shape
        """

        return self._shape

    @property
    def get_input_size(self) -> int:
        """
        Get input vector size for current network

        Returns
        -------
        size: int
            Input vector size
        """

        return self._input_size

    @property
    def get_output_size(self) -> int:
        """
        Get output vector size for current network

        Returns
        -------
        size: int
            Output vector size
        """

        return self._output_size

    @property
    def get_activations(self) -> list:
        """
        Get list of activations for each layer

        Returns
        -------
        activations: list
        """

        return self.network.get_activations

    def __str__(self) -> str:
        """
        Get a string representation of the neural network

        Returns
        -------
        result: str
        """

        return str(self.network)

    @classmethod
    def create_neuron(
        cls, input_size: int, output_size: int, shape: list[int], **kwargs
    ):
        """
        Create neural network with passed size and sigmoid activation

        Parameters
        ----------
        input_size: int
        output_size: int
        shape: list[int]
            Sizes of hidden layers
        kwargs

        Returns
        -------
        net: imodel.IModel
            Neural network
        """
        activation, decorator_params, weight, biases, kwargs = _get_act_and_init(
            kwargs,
            "sigmoid",
            None,
            tf.random_normal_initializer(),
        )

        res = cls(
            input_size=input_size,
            block_size=shape,
            output_size=output_size,
            activation_func=activation,
            bias_init=biases,
            weight_init=weight,
            decorator_params=decorator_params,
            **kwargs,
        )

        return res


_create_functions: defaultdict[str, Type[tf.keras.Model]] = defaultdict(
    lambda: TensorflowDenseNet
)
_create_functions["DenseNet"] = TensorflowDenseNet
