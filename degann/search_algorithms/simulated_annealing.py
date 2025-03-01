import math
import random
from typing import Optional, Tuple

from .nn_code import decode, default_alphabet
from degann.networks import imodel
from degann.search_algorithms.generate import (
    random_generate,
    choose_neighbor,
)
from .search_algorithms_parameters import SimulatedAnnealingSearchParameters
from .utils import update_random_generator, log_search_step


def simulated_annealing(
    parameters: SimulatedAnnealingSearchParameters,
) -> Tuple[float, int, str, str, dict, int]:
    """
    Method of simulated annealing in the parameter space of neural networks

    Parameters
    ----------
    parameters: SimulatedAnnealingSearchParameters
        Search algorithm parameters

    Returns
    -------
    search_results: tuple[float, int, str, str, dict]
        Results of the algorithm are described by these parameters

        best_metric_value: float
            The value of the loss function during training of the best neural network
        best_epoch: int
            Number of training epochs for the best neural network
        best_loss_func: str
            Name of the loss function of the best neural network
        best_opt: str
            Name of the optimizer of the best neural network
        best_net: dict
            Best neural network presented as a dictionary
        iterations: int
            The number of iterations performed during the search.
    """
    if parameters.nn_alphabet is None:
        parameters.nn_alphabet = default_alphabet

    gen = random_generate(
        min_epoch=parameters.min_epoch,
        max_epoch=parameters.max_epoch,
        min_length=parameters.nn_min_length,
        max_length=parameters.nn_max_length,
        alphabet=parameters.nn_alphabet,
        block_size=parameters.nn_alphabet_block_size,
    )
    if parameters.start_net is None:
        b, a = decode(
            gen[0].value(),
            block_size=parameters.nn_alphabet_block_size,
            offset=parameters.nn_alphabet_offset,
        )
        curr_best = imodel.IModel(
            parameters.input_size, b, parameters.output_size, a + ["linear"]
        )
    else:
        curr_best = imodel.IModel(
            parameters.input_size, [], parameters.output_size, ["linear"]
        )
        curr_best = curr_best.from_dict(parameters.start_net)

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
            parameters.val_data[0], parameters.val_data[1], verbose=0, return_dict=True
        )
        best_val_loss = val_metrics["loss"]
        best_val_metric_value = val_metrics[parameters.eval_metric]
    else:
        best_val_loss = None
        best_val_metric_value = None
    best_epoch = curr_epoch
    best_nn = curr_best.to_dict()
    best_gen = gen
    best_a = curr_best.get_activations
    best_metric_value = curr_metric_value

    if parameters.logging:
        fn = f"{parameters.file_name}_{len(parameters.data[0])}_0_{parameters.loss_function}_{parameters.optimizer}"
        log_search_step(
            model=curr_best,
            activations=best_a,
            code=best_gen[0].value(),
            epoch=best_gen[1].value(),
            optimizer=parameters.optimizer,
            loss_function=parameters.loss_function,
            loss=curr_loss,
            validation_loss=best_val_loss,
            metric_value=best_metric_value,
            validation_metric_value=best_val_metric_value,
            file_name=fn,
        )

    k = 0
    t = 1
    while (
        k < parameters.max_launches - 1
        and curr_metric_value > parameters.metric_threshold
    ):
        t = parameters.temperature_method(k=k, k_max=parameters.max_launches, t=t)
        print(parameters.distance_method)
        distance = parameters.distance_method(temperature=t)

        gen_neighbor = choose_neighbor(
            parameters.method_for_generate_next_nn,
            alphabet=parameters.nn_alphabet,
            block_size=parameters.nn_alphabet_block_size,
            parameters=(gen[0].value(), gen[1].value()),
            distance=distance,
            min_epoch=parameters.min_epoch,
            max_epoch=parameters.max_epoch,
            min_length=parameters.nn_min_length,
            max_length=parameters.nn_max_length,
        )
        b, a = decode(
            gen_neighbor[0].value(),
            block_size=parameters.nn_alphabet_block_size,
            offset=parameters.nn_alphabet_offset,
        )
        neighbor = imodel.IModel(
            parameters.input_size, b, parameters.output_size, a + ["linear"]
        )
        neighbor.compile(
            optimizer=parameters.optimizer,
            loss_func=parameters.loss_function,
            metrics=[parameters.eval_metric] + parameters.metrics,
        )
        neighbor_hist = neighbor.train(
            parameters.data[0],
            parameters.data[1],
            epochs=gen_neighbor[1].value(),
            verbose=0,
            callbacks=parameters.callbacks,
        )
        if parameters.val_data is not None:
            val_metrics = neighbor.evaluate(
                parameters.val_data[0],
                parameters.val_data[1],
                verbose=0,
                return_dict=True,
            )
            neighbor_val_loss = val_metrics[parameters.eval_metric]
            neighbor_val_metric_value = val_metrics["loss"]
        else:
            neighbor_val_loss = None
            neighbor_val_metric_value = None
        neighbor_loss = neighbor_hist.history["loss"][-1]
        neighbor_metric_value = neighbor_hist.history[parameters.eval_metric][-1]

        if (
            neighbor_metric_value < curr_metric_value
            or math.e ** ((curr_metric_value - neighbor_metric_value) / t)
            > random.random()
        ):
            curr_best = neighbor
            gen = gen_neighbor
            curr_epoch = gen_neighbor[1].value()
            curr_metric_value = neighbor_metric_value
            curr_val_metric_value = neighbor_val_metric_value

            if curr_metric_value < best_metric_value:
                best_metric_value = curr_metric_value
                best_epoch = curr_epoch
                best_nn = curr_best.to_dict()
                best_val_metric_value = curr_val_metric_value
        k += 1

        if parameters.logging:
            fn = f"{parameters.file_name}_{len(parameters.data[0])}_0_{parameters.loss_function}_{parameters.optimizer}"
            log_search_step(
                model=neighbor,
                activations=a,
                code=gen_neighbor[0].value(),
                epoch=gen_neighbor[1].value(),
                optimizer=parameters.optimizer,
                loss_function=parameters.loss_function,
                loss=neighbor_loss,
                validation_loss=neighbor_val_loss,
                metric_value=neighbor_metric_value,
                validation_metric_value=neighbor_val_metric_value,
                file_name=fn,
            )

    return (
        best_metric_value,
        best_epoch,
        parameters.loss_function,
        parameters.optimizer,
        best_nn,
        k,
    )
