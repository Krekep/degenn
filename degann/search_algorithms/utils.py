import csv
import random
from sys import stderr
from typing import Any, Optional, Callable

import numpy.random
import tensorflow as tf

from degann.networks import IModel

_algorithms_for_random_generator = {0: "auto_select", 1: "philox", 2: "threefry"}
tf.random.set_global_generator(numpy.random.default_rng())


def update_random_generator(curr_iter: int, cycle_size: int = 0) -> None:
    """
    Set global tensorflow random generator to random state every *cycle_size* times

    Parameters
    ----------
    curr_iter: int
        Counter showing whether it's time to update the random number generator
    cycle_size: int
        How often should we update random number generator (if not positive, then the generator does not change)

    Returns
    -------
    """
    if cycle_size > 0 and curr_iter % cycle_size == 0:
        new_g = tf.random.Generator.from_non_deterministic_state(
            alg=_algorithms_for_random_generator[
                random.randint(0, len(_algorithms_for_random_generator) - 1)
            ]
        )
        tf.random.set_global_generator(new_g)
    else:
        pass


def add_useless_argument(func: Callable) -> Callable:
    """
    Since Python can't hold function in attribute without bounding, this function simple add `self` argument to function

    Parameters
    ----------
    func: Callable

    Returns
    -------
    decorated_func: Callable
    """

    def inner(first: Any = None, *args, **kwargs):
        # print(f"*** Function {func.__name__}, first: {first}, args: {args}, kwargs: {kwargs}", file=stderr)
        return func(*args, **kwargs)

    return inner


def log_to_file(history: dict, fn: str) -> None:
    """
    Export history of training to file

    Parameters
    ----------
    history: dict
        History of training
    fn: str
        File name
    """
    with open(
        f"./{fn}.csv",
        "a",
        newline="",
    ) as outfile:
        writer = csv.writer(outfile)
        writer.writerows(zip(*history.values()))


def log_search_step(
    model: IModel,
    activations: list[str],
    epoch: int,
    optimizer: str,
    loss_function: str,
    loss: list[float],
    validation_loss: Optional[list[float]],
    metric_value: float,
    validation_metric_value: Optional[float],
    file_name: str,
) -> None:
    """
    Log model configuration and performance to file

    Parameters
    ----------
    model
    activations
    code
    epoch
    optimizer
    loss_function
    loss
    validation_loss
    metric_value
    validation_metric_value
    file_name

    Returns
    -------

    """
    history = SearchHistory()
    history.shapes = [model.get_shape]
    history.activations = [activations]
    history.epoch = [epoch]
    history.optimizer = [optimizer]
    history.loss_function = [loss_function]
    history.loss = [loss]
    history.validation_loss = [validation_loss]
    history.metric_value = [metric_value]
    history.validation_metric_value = [validation_metric_value]
    history.train_time = [model.network.trained_time["train_time"]]
    log_to_file(history.__dict__, file_name)


class SearchHistory:
    def __init__(self: "SearchHistory") -> None:
        self.shapes: list[list[int]]
        self.activations: list[list[str]]
        self.epoch: list[int]
        self.optimizer: list[str]
        self.loss_function: list[str]
        self.loss: list[float]
        self.validation_loss: list[Optional[float]]
        self.metric_value: list[float]
        self.validation_metric_value: list[Optional[float]]
        self.train_time: list[float]

    def __setitem__(self: "SearchHistory", __key: str, __value: Any):
        if __key in self.__dict__.keys():
            self.__dict__[__key] = __value
        else:
            raise ValueError(f"In SearchHistory there is no {__key} key")

    def __getitem__(self: "SearchHistory", __key: str) -> Any:
        return self.__dict__[__key]

    def __len__(self: "SearchHistory") -> int:
        return len(self.__dict__)

    def __iter__(self):
        return self.__dict__.__iter__()

    def keys(self):
        return self.__dict__.keys()
