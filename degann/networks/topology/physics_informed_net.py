from typing import List, Optional, Dict, Callable

import tensorflow as tf

# from tensorflow import keras
import keras

from degann.networks import layer_creator, losses, metrics, optimizers


class PhysicsInformedNet(keras.Model):
    def __init__(
        self,
        input_size: int = 2,
        block_size: list = None,
        output_size: int = 10,
        phys_func: Callable | None = None,
        boundary_func: Callable | None = None,
        phys_k=0.1,
        boundary_k=1.0,
        activation_func: str = "linear",
        weight=keras.initializers.RandomUniform(minval=-1, maxval=1),
        biases=keras.initializers.RandomUniform(minval=-1, maxval=1),
        layer: str | List[str] = "Dense",
        is_debug: bool = False,
        **kwargs,
    ):
        self._name = "PINN"
        decorator_params: List[Optional[Dict]] = [None]
        if "decorator_params" in kwargs.keys():
            decorator_params = kwargs.get("decorator_params")
            kwargs.pop("decorator_params")
        else:
            decorator_params = [None]

        if (
            isinstance(decorator_params, list)
            and len(decorator_params) == 1
            and decorator_params[0] is None
            or decorator_params is None
        ):
            decorator_params = [None] * (len(block_size) + 1)

        if (
            isinstance(decorator_params, list)
            and len(decorator_params) == 1
            and decorator_params[0] is not None
        ):
            decorator_params = decorator_params * (len(block_size) + 1)

        super(PhysicsInformedNet, self).__init__(**kwargs)
        self.blocks: List[keras.layers.Layer] = []

        if not isinstance(activation_func, list):
            activation_func = [activation_func] * (len(block_size) + 1)
        if not isinstance(layer, list):
            layer = [layer] * (len(block_size) + 1)
        if len(block_size) != 0:
            self.blocks.append(
                layer_creator.create(
                    input_size,
                    block_size[0],
                    activation=activation_func[0],
                    weight=weight,
                    layer_type=layer[0],
                    bias=biases,
                    is_debug=is_debug,
                    name=f"PINN0",
                    decorator_params=decorator_params[0],
                )
            )
            for i in range(1, len(block_size)):
                self.blocks.append(
                    layer_creator.create(
                        block_size[i - 1],
                        block_size[i],
                        activation=activation_func[i],
                        weight=weight,
                        bias=biases,
                        layer_type=layer[i],
                        is_debug=is_debug,
                        name=f"PINN{i}",
                        decorator_params=decorator_params[i],
                    )
                )
            last_block_size = block_size[-1]
        else:
            last_block_size = input_size

        self.out_layer = layer_creator.create(
            last_block_size,
            output_size,
            activation=activation_func[-1],
            weight=weight,
            bias=biases,
            layer_type=layer[-1],
            is_debug=is_debug,
            name=f"OutLayerPINN",
            decorator_params=decorator_params[-1],
        )

        self.activation_funcs = activation_func
        self.weight_initializer = weight
        self.bias_initializer = biases
        self.input_size = input_size
        self.block_size = block_size
        self.output_size = output_size
        self.trained_time = {"train_time": 0.0, "epoch_time": [], "predict_time": 0}
        self.phys_func = phys_func
        self.phys_k = phys_k
        self.boundary_func = boundary_func
        self.boundary_k = boundary_k

    @property
    def get_activations(self) -> List:
        """
        Get list of activations functions for each layer
        Returns
        -------
        activation: list
        """
        return [layer.get_activation for layer in self.blocks]

    def custom_compile(
        self,
        rate=1e-2,
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
        if metric_funcs is None:
            metric_funcs = []
        opt = optimizers.get_optimizer(optimizer)(learning_rate=rate)
        loss = losses.get_loss(loss_func)
        m = [metrics.get_metric(metric) for metric in metric_funcs]
        self.compile(
            optimizer=opt,
            loss=loss,
            metrics=m,
            run_eagerly=run_eagerly,
        )

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

    def train_step(self, data):
        """
        Custom train step with physics and
        boundary losses implementation
        Parameters
        ----------
        data: tuple
            Pair of x and y (or dataset)
        Returns
        -------
        """
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compute_loss(y=y, y_pred=y_pred)

            phys_deviation = self.phys_func(self, tape, x, y_pred)
            phys_loss = self.compiled_loss(
                tf.zeros_like(phys_deviation), phys_deviation
            )

            boundary_loss = 0
            if self.boundary_func is not None:
                boundary_deviation = self.boundary_func(self, tape, x, y_pred)
                boundary_loss = self.compiled_loss(
                    tf.zeros_like(boundary_deviation), boundary_deviation
                )

            total_loss = (
                loss + self.phys_k * phys_loss + self.boundary_k * boundary_loss
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        del tape
        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(total_loss)
            else:
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def set_name(self, new_name):
        self._name = new_name

    def __str__(self):
        res = f"IModel {self.name}\n"
        for layer in self.blocks:
            res += str(layer)
        res += str(self.out_layer)
        return res

    def to_dict(self, **kwargs):
        """
        Export neural network to dictionary
        Parameters
        ----------
        kwargs
        Returns
        -------
        """
        res = {
            "net_type": "MyPINN",
            "name": self._name,
            "input_size": self.input_size,
            "block_size": self.block_size,
            "output_size": self.output_size,
            "layer": [],
            "out_layer": self.out_layer.to_dict(),
        }

        for i, layer in enumerate(self.blocks):
            res["layer"].append(layer.to_dict())

        return res

    @classmethod
    def from_layers(
        cls,
        input_size: int,
        block_size: List[int],
        output_size: int,
        layers: List[keras.layers.Layer],
        **kwargs,
    ):
        """
        Restore neural network from list of layers
        Parameters
        ----------
        input_size
        block_size
        output_size
        layers
        kwargs
        Returns
        -------
        """
        res = cls(
            input_size=input_size,
            block_size=block_size,
            output_size=output_size,
            **kwargs,
        )

        for layer_num in range(len(res.blocks)):
            res.blocks[layer_num] = layers[layer_num]

        return res

    def from_dict(self, config, **kwargs):
        """
        Restore neural network from dictionary of params
        Parameters
        ----------
        config
        kwargs
        Returns
        -------
        """
        input_size = config["input_size"]
        block_size = config["block_size"]
        output_size = config["output_size"]

        self.block_size = list(block_size)
        self.input_size = input_size
        self.output_size = output_size

        layers: List[keras.layers.Layer] = []
        for layer_config in config["layer"]:
            layers.append(layer_creator.from_dict(layer_config))

        self.blocks: List[keras.layers.Layer] = []
        for layer_num in range(len(layers)):
            self.blocks.append(layers[layer_num])

        self.out_layer = layer_creator.from_dict(config["out_layer"])
