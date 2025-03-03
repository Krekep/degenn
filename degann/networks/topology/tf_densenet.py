import os
from typing import List, Optional, Dict

import tensorflow as tf

from degann.networks.config_format import LAYER_DICT_NAMES
from degann.networks import layer_creator, losses, metrics, cpp_utils
from degann.networks import optimizers
from degann.networks.layers.tf_dense import TensorflowDense
from degann.networks.topology.topology_parameters import TensorflowDenseNetParams
from degann.networks.topology.compile_parameters import SingleNetworkCompileParams


class TensorflowDenseNet(tf.keras.Model):
    def __init__(
        self, config: TensorflowDenseNetParams = TensorflowDenseNetParams(), **kwargs
    ):
        decorator_params: List[Optional[Dict]] = [None]
        if "decorator_params" in kwargs.keys():
            value = kwargs.get("decorator_params")
            if isinstance(value, list) and all(
                isinstance(item, dict) for item in value
            ):
                decorator_params = value
            kwargs.pop("decorator_params")
        else:
            decorator_params = [None]
        super(TensorflowDenseNet, self).__init__(**kwargs)

        if (
            isinstance(decorator_params, list)
            and len(decorator_params) == 1
            and decorator_params[0] is None
            or decorator_params is None
        ):
            decorator_params = [None] * (len(config.block_size) + 1)

        if (
            isinstance(decorator_params, list)
            and len(decorator_params) == 1
            and decorator_params[0] is not None
        ):
            decorator_params = decorator_params * (len(config.block_size) + 1)

        self.blocks: List[TensorflowDense] = []

        if not isinstance(config.activation_func, list):
            activation_func_list = [config.activation_func] * (
                len(config.block_size) + 1
            )
        else:
            activation_func_list = config.activation_func.copy()

        if len(config.block_size) != 0:
            self.blocks.append(
                layer_creator.create_dense(
                    config.input_size,
                    config.block_size[0],
                    activation=activation_func_list[0],
                    weight=config.weight,
                    bias=config.biases,
                    is_debug=config.is_debug,
                    name=f"TFDense0",
                    decorator_params=decorator_params[0],
                )
            )
            for i in range(1, len(config.block_size)):
                self.blocks.append(
                    layer_creator.create_dense(
                        config.block_size[i - 1],
                        config.block_size[i],
                        activation=activation_func_list[i],
                        weight=config.weight,
                        bias=config.biases,
                        is_debug=config.is_debug,
                        name=f"TFDense{i}",
                        decorator_params=decorator_params[i],
                    )
                )
            last_block_size = config.block_size[-1]
        else:
            last_block_size = config.input_size

        self.out_layer = layer_creator.create_dense(
            last_block_size,
            config.output_size,
            activation=activation_func_list[-1],
            weight=config.weight,
            bias=config.biases,
            is_debug=config.is_debug,
            name=f"OutLayerTFDense",
            decorator_params=decorator_params[-1],
        )

        self.activation_funcs = activation_func_list
        self.weight_initializer = config.weight
        self.bias_initializer = config.biases
        self.input_size = config.input_size
        self.block_size = config.block_size
        self.output_size = config.output_size
        self.trained_time = {"train_time": 0.0, "epoch_time": [], "predict_time": 0}

    def custom_compile(
        self, config: SingleNetworkCompileParams = SingleNetworkCompileParams()
    ):
        """
        Configures the model for training

        Parameters
        ----------
        config: SingleNetworkCompileParams
            parameters for compilation containing learning rate, optimizer,
            loss function and metrics

        Returns
        -------

        """
        opt = optimizers.get_optimizer(config.optimizer)(learning_rate=config.rate)
        loss = losses.get_loss(config.loss_func)
        m = [metrics.get_metric(metric) for metric in config.metric_funcs]
        self.compile(
            optimizer=opt,
            loss=loss,
            metrics=m,
            run_eagerly=config.run_eagerly,
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
        Custom train step from tensorflow tutorial

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
        with tf.GradientTape() as tape:
            y_pred: tf.Tensor = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compute_loss(y=y, y_pred=y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
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
            "net_type": "TFDense",
            # "name": self._name,
            "name": getattr(self, "_name", ""),
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
        layers: List[TensorflowDense],
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

    def from_dict(self, config: dict) -> None:
        """
        Restore neural network from dictionary of params

        Parameters
        ----------
        config: dict
            Model parameters

        """
        input_size = config["input_size"]
        block_size = config["block_size"]
        output_size = config["output_size"]

        self.block_size = list(block_size)
        self.input_size = input_size
        self.output_size = output_size

        layers: List[TensorflowDense] = []
        for layer_config in config["layer"]:
            layers.append(layer_creator.from_dict(layer_config))

        self.blocks.clear()
        for layer_num in range(len(layers)):
            self.blocks.append(layers[layer_num])

        self.out_layer = layer_creator.from_dict(config["out_layer"])

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
            path to c/c++ compiler
        kwargs

        Returns
        -------

        """
        res = """
#include <cmath>
#include <vector>

#define max(x, y) ((x < y) ? y : x)

        \n"""

        config = self.to_dict(**kwargs)

        input_size = self.input_size
        output_size = self.output_size
        blocks = self.block_size
        reverse = False
        layers = config["layer"] + [config["out_layer"]]

        comment = f"// This function takes {input_size} elements array and returns {output_size} elements array\n"
        signature = f""
        start_func = "{\n"
        end_func = "}\n"
        transform_input_vector = ""
        transform_output_array = ""
        return_stat = "return answer;\n"

        creator_1d = cpp_utils.array1d_creator("float")
        creator_heap_1d = cpp_utils.array1d_heap_creator("float")
        creator_2d = cpp_utils.array2d_creator("float")
        if array_type == "[]":
            signature = f"float* feedforward(float x_array[])\n"

        if array_type == "vector":
            signature = f"std::vector<float> feedforward(std::vector<float> x)\n"

            transform_input_vector = cpp_utils.transform_1dvector_to_array(
                "float", input_size, "x", "x_array"
            )
            transform_output_array = cpp_utils.transform_1darray_to_vector(
                "float", output_size, "answer_vector", "answer"
            )
            return_stat = "return answer_vector;\n"

        create_layers = ""
        create_layers += creator_1d(f"layer_0", input_size, 0)
        for i, size in enumerate(blocks):
            create_layers += creator_1d(f"layer_{i + 1}", size, 0)
        create_layers += creator_1d(f"layer_{len(blocks) + 1}", output_size, 0)
        create_layers += cpp_utils.copy_1darray_to_array(
            input_size, "x_array", "layer_0"
        )

        create_weights = ""
        for i, layer_dict in enumerate(layers):
            create_weights += creator_2d(
                f"weight_{i}_{i + 1}",
                layer_dict[LAYER_DICT_NAMES["inp_size"]],
                layer_dict[LAYER_DICT_NAMES["shape"]],
                layer_dict[LAYER_DICT_NAMES["weights"]],
                reverse,
            )

        fill_weights = ""

        create_biases = ""
        for i, layer_dict in enumerate(layers):
            create_biases += creator_1d(
                f"bias_{i + 1}",
                layer_dict[LAYER_DICT_NAMES["shape"]],
                layer_dict[LAYER_DICT_NAMES["bias"]],
            )

        fill_biases = ""
        feed_forward_cycles = ""
        for i, layer_dict in enumerate(layers):
            left_size = layer_dict[
                LAYER_DICT_NAMES["inp_size"]
            ]  # if i != 0 else input_size
            right_size = layer_dict[LAYER_DICT_NAMES["shape"]]
            act_func = layer_dict[LAYER_DICT_NAMES["activation"]]
            decorator_params = layer_dict.get(LAYER_DICT_NAMES["decorator_params"])
            feed_forward_cycles += cpp_utils.feed_forward_step(
                f"layer_{i}",
                left_size,
                f"layer_{i + 1}",
                right_size,
                f"weight_{i}_{i + 1}",
                f"bias_{i + 1}",
                act_func,
            )

        move_result = creator_heap_1d("answer", output_size)
        move_result += cpp_utils.copy_1darray_to_array(
            output_size, f"layer_{len(blocks) + 1}", "answer"
        )

        res += comment
        res += signature
        res += start_func
        res += transform_input_vector
        res += create_layers
        res += create_weights
        res += fill_weights
        res += create_biases
        res += fill_biases
        res += feed_forward_cycles
        res += move_result
        res += transform_output_array
        res += return_stat
        res += end_func

        header_res = f"""
        #ifndef {path[0].upper() + path[1:]}_hpp
        #define {path[0].upper() + path[1:]}_hpp
        #include "{path}.cpp"

        {comment}
        {signature};

        #endif /* {path[0].upper() + path[1:]}_hpp */

                """

        with open(path + ".cpp", "w") as f:
            f.write(res)

        with open(path + ".hpp", "w") as f:
            f.write(header_res)

        if path_to_compiler is not None:
            os.system(path_to_compiler + " -c -Ofast " + path + ".cpp")

    @property
    def get_activations(self) -> List:
        """
        Get list of activations functions for each layer

        Returns
        -------
        activation: list
        """
        return [layer.get_activation for layer in self.blocks]

    def get_loss_names(self):
        return ("loss",)
