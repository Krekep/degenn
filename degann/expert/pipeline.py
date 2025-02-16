from typing import Optional, List

from degann.expert import BaseParameters
from degann.search_algorithms import (
    random_search_endless,
    simulated_annealing,
    grid_search,
    generate_neighbor,
)
from degann.search_algorithms.search_algorithms_parameters import (
    BaseSearchParameters,
    RandomEarlyStoppingSearchParameters,
    SimulatedAnnealingSearchParameters,
    GridSearchParameters,
)


def execute_pipeline(
    input_size: int,
    output_size: int,
    data: tuple,
    parameters: BaseParameters,
    run_grid_search: bool = False,
    additional_losses: Optional[List[str]] = None,
    additional_optimizers: Optional[List[str]] = None,
    val_data=None,
    **kwargs,
) -> tuple[float, dict]:
    """
    This function sequentially launches algorithms for searching the topology of a neural network
    with the passed parameters and returns the resulting neural network.

    Parameters
    ----------
    input_size: int
        Feature vector size
    output_size: int
        Value vector size
    data: tuple
        Dataset
    parameters: BaseParameters
        Parameters for search algorithms
    values: dict
        Parameters for creating and training neural networks
    run_grid_search: bool
        If `True`, then if the random search and the simulated annealing method fail, the grid search will be launched
    additional_losses: Optional[List[str]]
        Additional losses for grid search
    additional_optimizers: Optional[List[str]]
        Additional optimizers for grid search
    val_data: tuple
        Validation dataset
    kwargs

    Returns
    -------
    search_result: tuple[float, dict]
        Loss value and resulting neural network
    """
    if additional_losses is None:
        additional_losses = list()
    if additional_optimizers is None:
        additional_optimizers = list()

    search_alg_params = BaseSearchParameters()
    search_alg_params.loss_function = parameters.loss_function
    search_alg_params.eval_metric = parameters.eval_metric
    search_alg_params.optimizer = parameters.optimizer
    search_alg_params.input_size = input_size
    search_alg_params.output_size = output_size
    search_alg_params.data = data
    search_alg_params.val_data = val_data
    search_alg_params.min_epoch = parameters.min_train_epoch
    search_alg_params.max_epoch = parameters.max_train_epoch
    search_alg_params.nn_min_length = parameters.nn_min_length
    search_alg_params.nn_max_length = parameters.nn_max_length
    search_alg_params.nn_alphabet = parameters.nn_alphabet
    search_alg_params.nn_alphabet_block_size = parameters.nn_alphabet_block_size
    search_alg_params.nn_alphabet_offset = parameters.nn_alphabet_offset

    best_metric_value, best_nn = 1e6, dict()
    for i in range(parameters.launch_count_random_search):
        random_search_parameters = RandomEarlyStoppingSearchParameters(
            search_alg_params
        )
        random_search_parameters.metric_threshold = parameters.metric_threshold
        random_search_parameters.max_launches = parameters.iteration_count
        random_search_parameters.iterations = 1
        (
            train_metric_value,
            count_epoch,
            loss_function,
            optimizer,
            result_nn,
            last_iteration,
        ) = random_search_endless(
            random_search_parameters,
            verbose=True,
        )
        print(f"Ended {i} launch of random search")
        if train_metric_value <= random_search_parameters.metric_threshold:
            return train_metric_value, result_nn
        if train_metric_value <= best_metric_value:
            best_metric_value = train_metric_value
            best_nn = result_nn
    print("Random search didn't find any results")

    for i in range(parameters.launch_count_simulated_annealing):
        simulated_annealing_parameters = SimulatedAnnealingSearchParameters(
            search_alg_params
        )
        simulated_annealing_parameters.metric_threshold = parameters.metric_threshold
        simulated_annealing_parameters.max_launches = parameters.iteration_count
        simulated_annealing_parameters.iterations = 1
        simulated_annealing_parameters.method_for_generate_next_nn = generate_neighbor
        simulated_annealing_parameters.temperature_method = (
            parameters.simulated_annealing_params.temperature_reduction_method(
                parameters.simulated_annealing_params.temperature_speed
            )
        )
        simulated_annealing_parameters.distance_method = (
            parameters.simulated_annealing_params.distance_to_neighbor(
                parameters.simulated_annealing_params.dist_offset,
                multiplier=parameters.simulated_annealing_params.dist_scale,
            )
        )
        (
            train_metric_value,
            count_epoch,
            loss_function,
            optimizer,
            result_nn,
            last_iteration,
        ) = simulated_annealing(simulated_annealing_parameters)
        print(f"Ended {i} launch of SAM")
        if train_metric_value <= simulated_annealing_parameters.metric_threshold:
            return train_metric_value, result_nn
        if train_metric_value <= best_metric_value:
            best_metric_value = train_metric_value
            best_nn = result_nn
    print("Simulated annealing didn't find any results")

    if run_grid_search:
        grid_search_parameters = GridSearchParameters(search_alg_params)
        grid_search_parameters.losses = [
            search_alg_params.loss_function
        ] + additional_losses
        grid_search_parameters.optimizers = [
            search_alg_params.optimizer
        ] + additional_optimizers
        grid_search_parameters.epoch_step = 50
        (
            train_metric_value,
            count_epoch,
            loss_function,
            optimizer,
            result_nn,
        ) = grid_search(grid_search_parameters)

        return train_metric_value, result_nn

    return best_metric_value, best_nn
