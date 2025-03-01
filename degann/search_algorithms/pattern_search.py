import random
from itertools import product
from typing import Optional, Union, Any, Tuple, List, Iterable, Sized, Collection

import numpy as np
import tensorflow as tf

from degann.networks.callbacks import MemoryCleaner
from degann.networks import get_all_optimizers, get_all_metric_functions
from degann.networks import activations, imodel, losses

_default_shapes = [
    [10, 10, 10, 10, 10, 10],
    [80, 80, 80],
    [32, 16, 8, 4],
    [4, 8, 16, 32],
    [10, 10],
    [30],
]


def _create_random_network(
    min_layers=1,
    max_layers=5,
    min_neurons=3,
    max_neurons=60,
) -> tuple[list[int], list[str], list[None]]:
    """
    Create random neural network from the passed parameters.

    Parameters
    ----------
    min_layers: int
        Minimal count of layers in neural net
    max_layers: int
        Maximal count of layers in neural net
    min_neurons: int
        Minimal count of neurons per layer
    max_neurons: int
        Maximal count of neurons per layer
    Returns
    -------
    net: tuple[list[int], list[str], list[None]]
        Random neural network
    """

    layers = random.randint(min_layers, max_layers)
    shape = [random.randint(min_neurons, max_neurons) for _ in range(layers)]
    act = []
    decorator_param: list[None] = []
    all_act_names = list(activations.get_all_activations().keys())
    for _ in shape:
        act.append(random.choice(all_act_names))
        # TODO: activation func can take additional arguments
        # but at this moment I dont create random arguments (instead of *None* in decorator_params)
        decorator_param.append(None)
    act.append("linear")
    decorator_param.append(None)

    nets_param = (shape, act, decorator_param)
    return nets_param


def _normalize_array(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Scale array from [a, b] to [0, 1]

    Parameters
    ----------
    x: np.ndarray
        Array for scaling

    Returns
    -------
    x: Tuple[np.ndarray, float]
        Scaled array and scale coefficient
    """
    m = abs(np.amax(x))
    if m != 0:
        x = x / m
    return x, m


def train(
    x_data: np.ndarray,
    y_data: np.ndarray,
    val_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    **kwargs,
) -> tuple[imodel.IModel, dict[Union[str, Any], Any]]:
    """
    Choose and return neural network which present the minimal average absolute deviation.
    x_data and y_data is numpy 2d arrays (in case we don't have multiple-layer input/output).

    Parameters
    ----------
    x_data: np.ndarray
        Array of inputs --- [input1, input2, ...]
    y_data: np.ndarray
        List of outputs --- [output1, output2, ...]
    val_data: np.ndarray
        Validation dataset

    Returns
    -------
    net: imodel.IModel
        Best neural network for this dataset
    history: Dict[str, list]
        History of training for this network
    """
    # default config
    args = TrainConfig()
    for kw in kwargs:
        if kw in args.keys():
            args[kw] = kwargs[kw]

    if args.debug:
        print("Start train func")

    # determining the number of inputs and outputs of the neural network
    if type(x_data[0]) is np.ndarray:
        input_len = len(x_data[0])
    else:
        input_len = 1

    if type(y_data[0]) is np.ndarray:
        output_len = len(y_data[0])
    else:
        output_len = 1

    # prepare data (normalize)
    norm_coff: float = 1
    if args.normalize:
        x_data, norm_coff = _normalize_array(x_data)
        # y_data, norm_coff = _normalize_array(y_data)
        if args.debug:
            print(f"Normalization coefficient is {norm_coff}")

    x_data_tensor = tf.convert_to_tensor(x_data, dtype=tf.float32)
    y_data_tensor = tf.convert_to_tensor(y_data, dtype=tf.float32)

    # prepare neural networks
    if args.debug:
        print("Prepare neural networks and data")
    nets = []
    for parameters in args.nets_param:
        shape: list[int] = parameters[0]
        act = parameters[1]
        decorator_param = parameters[2]
        str_shape = "_".join(map(str, shape))
        curr_net = imodel.IModel(
            input_size=input_len,
            block_size=shape,
            output_size=output_len,
            activation_func=act,
            decorator_params=decorator_param,
            net_type=args.net_type,
            name=f"net{args.name_salt}_{str_shape}",
            is_debug=args.debug,
        )
        nets.append(curr_net)
    if args.use_rand_net:
        rand_net_params = _create_random_network(input_len, output_len)
        str_shape = "_".join(map(str, rand_net_params[0]))
        rand_net = imodel.IModel(
            input_size=input_len,
            block_size=rand_net_params[0],
            output_size=output_len,
            activation_func=rand_net_params[1],
            decorator_params=rand_net_params[2],
            net_type=args.net_type,
            name=f"net{args.name_salt}_{str_shape}",
        )
        nets.append(rand_net)

    # compile
    for nn in nets:
        nn.compile(
            rate=args.eps,
            optimizer=args.optimizer,
            loss_func=args.loss_function,
            metrics=[args.eval_metric] + args.metrics,
            # run_eagerly=True,
        )

    if args.debug:
        print("Success prepared")

    # train
    history = []
    for i, nn in enumerate(nets):
        verb = 0
        if args.debug:
            print(nn)
            verb = 1
        temp_his = nn.train(
            x_data_tensor,
            y_data_tensor,
            epochs=args.epochs,
            validation_data=val_data,
            callbacks=[MemoryCleaner()],
            verbose=verb,
        )
        temp_last_res = dict()
        for key in temp_his.history:
            temp_last_res[key] = temp_his.history[key].copy()

        history.append(temp_last_res)
    result_net = nets[0]
    result_history = history[0]
    min_err = history[0][args.eval_metric]
    for i in range(1, len(nets)):
        if history[i][args.eval_metric] < min_err:
            min_err = history[i][args.eval_metric]
            result_net = nets[i]
            result_history = history[i]
    if args.debug:
        print(f"Minimal metric error is {min_err} {args.name_salt}")
    return result_net, result_history


def pattern_search(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    **kwargs,
) -> list[tuple[dict, float, float, imodel.IModel]]:
    """
    Choose and return neural network which present the minimal average absolute deviation.
    x_data and y_data is numpy 2d arrays (in case we don't have multiple-layer input/output).

    Parameters
    ----------
    x_data: np.ndarray
        Array of inputs --- [input1, input2, ...]
    y_data: np.ndarray
        List of outputs --- [output1, output2, ...]
    x_val: np.ndarray
        Array of inputs for validate train
    y_val: np.ndarray
        Array of outputs for validate train
    Returns
    -------
    net: network.INetwork
        Best neural network for this dataset
    """

    # default config
    args = PatternSearchConfig()

    for arg in args:
        if kwargs.get(arg) is not None:
            args[arg] = kwargs[arg]
            kwargs.pop(arg)

    val_data = None
    if x_val is not None and y_val is not None:
        val_data = (x_val, y_val)

    # Networks parameters --- shape and activation functions
    nets_param = []
    for shape in args.net_shapes:
        if len(shape) != 0:
            for activation in args.activations:
                nets_param.append(
                    [
                        shape,
                        [activation] * len(shape) + ["linear"],
                        [None] * (len(shape) + 1),
                    ]
                )
        else:
            nets_param.append(
                [
                    shape,
                    ["linear"],
                    [None],
                ]
            )

    # too long for mypy checking
    # metaparams = list(map(lambda x: dict(x) | kwargs,product(*list(map(lambda x: [x] if isinstance(x[0], str) else x,(list(map(lambda kv: ([(kv[0], v) for v in kv[1]]if isinstance(kv[1], (Iterable, Sized))and len(kv[1]) > 0else (kv[0], kv[1])), args.__dict__.items()))))))))

    list_of_decomposed_args = list(
        map(
            lambda kv: (
                [(kv[0], v) for v in kv[1]]
                if isinstance(kv[1], list) and len(kv[1]) > 0
                else (kv[0], kv[1])
            ),
            args.__dict__.items(),
        )
    )
    all_args_combinations = product(
        *list(
            map(
                lambda x: [x] if isinstance(x[0], str) else x,
                list_of_decomposed_args,
            )
        )
    )
    metaparams = list(map(lambda x: dict(x) | kwargs, all_args_combinations))

    best_nets: List[tuple[dict, float, float, imodel.IModel]] = []
    if kwargs.get("debug"):
        print(f"Grid search size {len(metaparams)}")
        print("Amount of networks for each set of parameters", len(nets_param))
    for i, params in enumerate(metaparams):
        if kwargs.get("debug"):
            print(f"Number of set {i}")
        trained, history = train(x_data, y_data, val_data=val_data, **params)
        metric_value = history[args.eval_metric][-1]
        if val_data is not None:
            val_metric_value = history["val_" + args.eval_metric][-1]
        else:
            val_metric_value = 10**9
        best_nets.append((params, metric_value, val_metric_value, trained))

    best_nets.sort(key=lambda x: [x[1], x[2]])
    return best_nets


class PatternSearchConfig:
    def __init__(self: "PatternSearchConfig") -> None:
        self.loss_functions: list[str] = [
            key for key in losses.get_all_loss_functions()
        ]
        self.optimizers: list[str] = [key for key in get_all_optimizers()]
        self.metrics: list[list[str]] = [[key for key in get_all_metric_functions()]]
        self.eval_metric: str = "root_mean_squared_error"
        self.net_shapes: list[list[int]] = _default_shapes
        self.activations: list[str] = [key for key in activations.get_all_activations()]
        self.rates: list[float] = [1e-2, 5e-3, 1e-3]
        self.epochs: list[int] = [50, 200]

    def __setitem__(self: "PatternSearchConfig", __key: str, __value: Any):
        if __key in self.__dict__.keys():
            self.__dict__[__key] = __value
        else:
            raise ValueError(f"In PatternSearchConfig there is no {__key} key")

    def __getitem__(self: "PatternSearchConfig", __key: str) -> Any:
        return self.__dict__[__key]

    def __len__(self: "PatternSearchConfig") -> int:
        return len(self.__dict__)

    def __iter__(self):
        return self.__dict__.__iter__()

    def keys(self):
        return self.__dict__.keys()


class TrainConfig:
    def __init__(self: "TrainConfig") -> None:
        self.debug: bool = False
        self.eps: float = 1e-2
        self.epochs: int = 100
        self.normalize: bool = False
        self.name_salt: str = ""
        self.loss_function: str = "MeanSquaredError"
        self.optimizer: str = "SGD"
        self.metrics: list[str] = []
        self.eval_metric: str = "root_mean_squared_error"
        self.use_rand_net: bool = True
        self.net_type: str = "DenseNet"
        self.nets_param: list[tuple[list[int], list[str], list[None]]] = [
            (
                shape,  # shape
                ["sigmoid"] * len(shape) + ["linear"],  # activation functions
                [None] * (len(shape) + 1),  # decorator parameters for activation
            )
            for shape in _default_shapes
        ]

    def __setitem__(self: "TrainConfig", __key: str, __value: Any) -> None:
        if __key in self.__dict__.keys():
            self.__dict__[__key] = __value
        else:
            raise ValueError(f"In TrainConfig there is no {__key} key")

    def __getitem__(self: "TrainConfig", __key: str) -> Any:
        return self.__dict__[__key]

    def __len__(self: "TrainConfig") -> int:
        return len(self.__dict__)

    def __iter__(self):
        return self.__dict__.__iter__()

    def keys(self):
        return self.__dict__.keys()
