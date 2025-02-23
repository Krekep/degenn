from typing import Optional, Tuple

from .nn_code import decode, default_alphabet
from degann.networks import imodel
from degann.search_algorithms.generate import random_generate
from .search_algorithms_parameters import (
    RandomEarlyStoppingSearchParameters,
    RandomSearchParameters,
)
from .utils import update_random_generator, log_search_step


def random_search(
    parameters: RandomSearchParameters,
) -> Tuple[float, int, str, str, dict]:
    """
    Algorithm for random search in the space of parameters of neural networks

    Parameters
    ----------
    parameters: RandomSearchParameters
        Search algorithm parameters

    Returns
    -------
    search_results: tuple[float, int, str, str, dict]
        Results of the algorithm are described by these parameters

        best_metric_value: float
            The value of the metric during training of the best neural network
        best_epoch: int
            Number of training epochs for the best neural network
        best_loss_func: str
            Name of the loss function of the best neural network
        best_opt: str
            Name of the optimizer of the best neural network
        best_net: dict
            Best neural network presented as a dictionary
    """
    if parameters.nn_alphabet is None:
        parameters.nn_alphabet = default_alphabet

    best_net: dict
    best_metric_value = 1e6
    best_epoch: int

    assert parameters.iterations > 0, "The number of iterations must be positive."

    for i in range(parameters.iterations):
        gen = random_generate(
            min_epoch=parameters.min_epoch,
            max_epoch=parameters.max_epoch,
            min_length=parameters.nn_min_length,
            max_length=parameters.nn_max_length,
            alphabet=parameters.nn_alphabet,
            block_size=parameters.nn_alphabet_block_size,
        )

        b, a = decode(
            gen[0].value(),
            block_size=parameters.nn_alphabet_block_size,
            offset=parameters.nn_alphabet_offset,
        )
        curr_best = imodel.IModel(
            parameters.input_size, b, parameters.output_size, a + ["linear"]
        )
        curr_best.compile(
            optimizer=parameters.optimizer,
            loss_func=parameters.loss_function,
            metrics=[parameters.eval_metric] + parameters.metrics,
        )
        curr_epoch = gen[1].value()
        hist = curr_best.train(
            parameters.data[0],
            parameters.data[1],
            epochs=curr_epoch,
            verbose=0,
            callbacks=parameters.callbacks,
        )
        curr_loss = hist.history["loss"][-1]
        curr_metric_value = hist.history[parameters.eval_metric][-1]
        if parameters.val_data is not None:
            val_metrics = curr_best.evaluate(
                parameters.val_data[0],
                parameters.val_data[1],
                verbose=0,
                return_dict=True,
            )
            curr_val_loss = val_metrics["loss"]
            curr_val_metric_value = val_metrics[parameters.eval_metric]
        else:
            curr_val_loss = None
            curr_val_metric_value = None

        if parameters.logging:
            fn = f"{parameters.file_name}_{len(parameters.data[0])}_0_{parameters.loss_function}_{parameters.optimizer}"
            log_search_step(
                model=curr_best,
                activations=a,
                code=gen[0].value(),
                epoch=gen[1].value(),
                optimizer=parameters.optimizer,
                loss_function=parameters.loss_function,
                loss=curr_loss,
                validation_loss=curr_val_loss,
                metric_value=curr_metric_value,
                validation_metric_value=curr_val_metric_value,
                file_name=fn,
            )

        if curr_metric_value < best_metric_value:
            best_epoch = curr_epoch
            best_net = curr_best.to_dict()
            best_metric_value = curr_metric_value
    return (
        best_metric_value,
        best_epoch,
        parameters.loss_function,
        parameters.optimizer,
        best_net,
    )


def random_search_endless(
    parameters: RandomEarlyStoppingSearchParameters, verbose: bool = False
) -> Tuple[float, int, str, str, dict, int]:
    """
    Algorithm for random search in the space of parameters of neural networks

    Parameters
    ----------
    parameters: RandomEarlyStoppingSearchParameters
        Search algorithm parameters
    verbose: bool
        If True, it will show additional information when searching

    Returns
    -------
    search_results: tuple[float, int, str, str, dict, int]
        Results of the algorithm are described by these parameters

        best_metric_value: float
            The value of the metric during training of the best neural network
        best_epoch: int
            Number of training epochs for the best neural network
        best_loss_func: str
            Name of the loss function of the best neural network
        best_opt: str
            Name of the optimizer of the best neural network
        best_net: dict
            Best neural network presented as a dictionary
        last_iter: int
            Count of iterations in search algorithm
    """
    if parameters.nn_alphabet is None:
        parameters.nn_alphabet = default_alphabet

    nn_metric_value, nn_epoch, loss_f, opt_n, net = random_search(parameters)
    i = 1
    best_net = net
    best_metric_value = nn_metric_value
    best_epoch = nn_epoch
    while (
        nn_metric_value > parameters.metric_threshold and i != parameters.max_launches
    ):
        if verbose:
            print(
                f"Random search until less than threshold. Last loss = {nn_metric_value}. Iterations = {i}"
            )
        nn_metric_value, nn_epoch, loss_f, opt_n, net = random_search(parameters)
        i += 1
        if nn_metric_value < best_metric_value:
            best_net = net
            best_metric_value = nn_metric_value
            best_epoch = nn_epoch
    return (
        best_metric_value,
        best_epoch,
        parameters.loss_function,
        parameters.optimizer,
        best_net,
        i,
    )
