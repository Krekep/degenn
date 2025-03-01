from typing import Optional, List, Callable

import numpy as np

from degann.search_algorithms.generate import generate_neighbor
from degann.search_algorithms.nn_code import default_alphabet
import numpy.typing as npt

from degann.search_algorithms.simulated_annealing_functions import (
    temperature_lin,
    distance_const,
)
from degann.search_algorithms.utils import add_useless_argument


class BaseSearchParameters:
    """
    Basic parameters of all model topology search algorithms

    Attributes
    ----------
    input_size: int
       Size of input data
    output_size: int
        Size of output data
    data: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        dataset
    val_data: Optional[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]
        Validation dataset
    min_epoch: int
        Lower bound of epochs
    max_epoch: int
        Upper bound of epochs
    optimizer: str
        Name of optimizer
    loss_function: str
        Name of loss function
    nn_min_length: int
        Starting number of hidden layers of neural networks
    nn_max_length: int
        Final number of hidden layers of neural networks
    nn_alphabet: Optional[list[str]]
        List of possible sizes of hidden layers with activations for them
    nn_alphabet_block_size: int
        Number of literals in each `alphabet` symbol that indicate the size of hidden layer
    nn_alphabet_offset: int
        Indicate the minimal number of neurons in hidden layer
    callbacks: list
        Callbacks for neural networks training
    file_name: str
        Path to file for logging
    logging: bool
        Logging search process to file
    """

    __slots__ = [
        "input_size",
        "output_size",
        "data",
        "val_data",
        "nn_max_length",
        "nn_min_length",
        "nn_alphabet_block_size",
        "nn_alphabet_offset",
        "nn_alphabet",
        "min_epoch",
        "max_epoch",
        "loss_function",
        "optimizer",
        "callbacks",
        "logging",
        "file_name",
        "metrics",
        "eval_metric",
    ]

    def __init__(self) -> None:
        self.input_size: int
        self.output_size: int
        self.data: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        self.val_data: Optional[
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        ] = None
        self.nn_max_length: int = 4
        self.nn_min_length: int = 1
        self.nn_alphabet_block_size: int = 1
        self.nn_alphabet_offset: int = 8
        self.nn_alphabet: Optional[list[str]] = default_alphabet
        self.min_epoch: int = 200
        self.max_epoch: int = 500
        self.loss_function: str = "MSE"
        self.optimizer: str = "Adam"
        self.callbacks: Optional[list] = None
        self.logging: bool = False
        self.file_name: str = ""
        self.metrics: list[str] = []
        self.eval_metric: str = "root_mean_squared_error"

    def fill_from_other(self, other: "BaseSearchParameters"):
        self.input_size = other.input_size
        self.output_size = other.output_size
        self.data = other.data
        self.val_data = other.val_data
        self.nn_max_length = other.nn_max_length
        self.nn_min_length = other.nn_min_length
        self.nn_alphabet_block_size = other.nn_alphabet_block_size
        self.nn_alphabet = other.nn_alphabet
        self.min_epoch = other.min_epoch
        self.max_epoch = other.max_epoch
        self.loss_function = other.loss_function
        self.optimizer = other.optimizer
        self.callbacks = other.callbacks
        self.logging = other.logging
        self.file_name = other.file_name
        self.metrics = other.metrics
        self.eval_metric = other.eval_metric


class GridSearchParameters(BaseSearchParameters):
    """
    Parameters of grid algorithm for search model topology

    Attributes
    ----------
    epoch_step: int
        Step between `min_epoch` and `max_epoch`
    optimizers: list[str]
        List of optimizers
    loss: list[str]
        list of loss functions
    """

    __slots__ = ["epoch_step", "optimizers", "losses"]

    def __init__(self, parent: BaseSearchParameters) -> None:
        super().__init__()
        self.fill_from_other(parent)
        self.epoch_step: int = 50
        self.optimizers: List[str]
        self.losses: List[str]


class RandomSearchParameters(BaseSearchParameters):
    """
    Parameters of random algorithm for search model topology

    Attributes
    ----------
    iterations: int
        The number of iterations that will be carried out within the algorithm before completion
        (specifically, the number of trained neural networks)
    """

    __slots__ = ["iterations"]

    def __init__(self, parent: BaseSearchParameters) -> None:
        super().__init__()
        self.fill_from_other(parent)
        self.iterations: int


class RandomEarlyStoppingSearchParameters(RandomSearchParameters):
    """
    Parameters of condition random algorithm for search model topology

    Attributes
    ----------
    loss_threshold: float
        Training will stop when the value of the loss function is less than this threshold
    max_launches: int
        Training will stop when the number of iterations of the algorithm exceeds this parameter
    """

    __slots__ = ["max_launches", "metric_threshold"]

    def __init__(self, parent: BaseSearchParameters) -> None:
        super().__init__(parent)
        self.max_launches: int  # -1 for endless search. Number of trained networks equals to `max_launches` * `iterations`
        self.metric_threshold: float


class SimulatedAnnealingSearchParameters(RandomEarlyStoppingSearchParameters):
    """
    Parameters of condition random algorithm for search model topology

    Attributes
    ----------
    start_net: dict
        Start point in parameter space of neural networks, if None it will be random generated
    method_for_generate_next_nn: Callable
        Method for obtaining the next point in parameter space of neural networks
    temperature_method: Callable
        Temperature decreasing method in SAM
    distance_method: Callable
        Method that sets the boundaries of the neighborhood around the current point
    """

    __slots__ = [
        "start_net",
        "method_for_generate_next_nn",
        "temperature_method",
        "distance_method",
    ]

    # iterations doesn't matter in this search algorithm
    def __init__(self, parent: BaseSearchParameters) -> None:
        super().__init__(parent)
        self.start_net: Optional[dict] = None
        self.method_for_generate_next_nn: Callable = add_useless_argument(
            generate_neighbor
        )
        self.temperature_method: Callable = add_useless_argument(temperature_lin)
        self.distance_method: Callable = add_useless_argument(distance_const(150))
