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

from degann.networks.topology.topology_parameters import BaseTopologyParams
from degann.networks.topology.compile_parameters import BaseCompileParams


class BaseSearchParameters:
    """
    Basic parameters of all model topology search algorithms

    Attributes
    ----------
    model_cfg: BaseTopologyParams
        Configurable model with `tunable_field` attributes
    compile_cfg: BaseCompileParams
        Compile config with `tunable_field` attributes
    data: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        dataset
    val_data: Optional[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]
        Validation dataset
    min_epoch: int
        Lower bound of epochs
    max_epoch: int
        Upper bound of epochs
    callbacks: list
        Callbacks for neural networks training
    file_name: str
        Path to file for logging
    logging: bool
        Logging search process to file
    """

    __slots__ = [
        "model_cfg",
        "compile_cfg",
        "data",
        "val_data",
        "min_epoch",
        "max_epoch",
        "callbacks",
        "logging",
        "file_name",
        "eval_metric",
    ]

    def __init__(self) -> None:
        self.model_cfg: BaseTopologyParams
        self.compile_cfg: BaseCompileParams
        self.data: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        self.val_data: Optional[
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        ] = None
        self.min_epoch: int = 200
        self.max_epoch: int = 500
        self.callbacks: Optional[list] = None
        self.logging: bool = False
        self.file_name: str = ""
        self.eval_metric: str = "root_mean_squared_error"

    def fill_from_other(self, other: "BaseSearchParameters"):
        self.model_cfg = other.model_cfg
        self.compile_cfg = other.compile_cfg
        self.data = other.data
        self.val_data = other.val_data
        self.min_epoch = other.min_epoch
        self.max_epoch = other.max_epoch
        self.callbacks = other.callbacks
        self.logging = other.logging
        self.file_name = other.file_name
        self.eval_metric = other.eval_metric


class GridSearchParameters(BaseSearchParameters):
    """
    Parameters of grid algorithm for search model topology

    Attributes
    ----------
    epoch_step: int
        Step between `min_epoch` and `max_epoch`
    """

    __slots__ = ["epoch_step"]

    def __init__(self, parent: BaseSearchParameters) -> None:
        super().__init__()
        self.fill_from_other(parent)
        self.epoch_step: int = 50


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
    method_for_generate_next_nn: Callable
        Method for obtaining the next point in parameter space of neural networks
    temperature_method: Callable
        Temperature decreasing method in SAM
    distance_method: Callable
        Method that sets the boundaries of the neighborhood around the current point
    """

    __slots__ = [
        "method_for_generate_next_nn",
        "temperature_method",
        "distance_method",
    ]

    # iterations doesn't matter in this search algorithm
    def __init__(self, parent: BaseSearchParameters) -> None:
        super().__init__(parent)
        self.method_for_generate_next_nn: Callable = add_useless_argument(
            generate_neighbor
        )
        self.temperature_method: Callable = add_useless_argument(temperature_lin)
        self.distance_method: Callable = add_useless_argument(distance_const(150))
