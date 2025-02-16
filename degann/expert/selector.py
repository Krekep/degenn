from degann.expert.tags import (
    EquationType,
    ExpertSystemTags,
    DataSize,
    ModelPredictTime,
    RequiredModelPrecision,
)
from degann.search_algorithms.nn_code import default_alphabet
from typing import Optional, Callable

from degann.search_algorithms.simulated_annealing_functions import (
    distance_lin,
    distance_const,
    temperature_lin,
    temperature_exp,
)


class BaseSamParameters:
    distance_to_neighbor: Callable = distance_const
    dist_offset: int = 300
    dist_scale: int = 0
    temperature_reduction_method: Callable = temperature_lin
    temperature_speed: float = 0


class BaseParameters:
    launch_count_random_search: int = 2
    launch_count_simulated_annealing: int = 2
    nn_max_length: int = 4
    nn_min_length: int = 1
    nn_alphabet_block_size: int = 1
    nn_alphabet_offset: int = 8
    nn_alphabet: list[str] = default_alphabet
    min_train_epoch: int = 200
    max_train_epoch: int = 500
    iteration_count: int = 5
    loss_function: str = "MaxAbsoluteDeviation"
    eval_metric: str = "root_mean_squared_error"
    metric_threshold: float = 1
    optimizer: str = "Adam"
    simulated_annealing_params: BaseSamParameters = BaseSamParameters()


def suggest_parameters(
    data: Optional[tuple] = None,
    tags: Optional[ExpertSystemTags] = None,
) -> BaseParameters:
    """
    Builds many parameters of search algorithms using labels supplied by the user,
     describing the requirements for the result and hints on the data.

    Parameters
    ----------
    data: Optional[tuple]
        Dataset
    tags: dict[ExpertSystemTags]
        A subset of tags described in expert_system_tags

    Returns
    -------
    parameters: BaseParameters
        Parameters for search algorithms
    """
    if tags is None:
        tags = ExpertSystemTags()
        tags.equation_type = EquationType.UNKNOWN
        tags.model_precision = RequiredModelPrecision.MAXIMAL
        tags.predict_time = ModelPredictTime.LONG
        tags.data_size = DataSize.AUTO

    parameters = BaseParameters()

    simulated_annealing_params = BaseSamParameters()

    if tags.equation_type in [
        EquationType.SIN,
        EquationType.MULTIDIM,
        EquationType.UNKNOWN,
    ]:
        parameters.min_train_epoch *= 2
        parameters.max_train_epoch = 700
        parameters.nn_max_length += 1
        parameters.iteration_count += 10

        # simulated_annealing_params["distance_to_neighbor"] = [distance_const(300), distance_lin(50, 400)]
        # simulated_annealing_params["temperature_reduction_method"] = [temperature_exp(0.95), temperature_exp(0.95)]
        simulated_annealing_params.distance_to_neighbor = distance_lin
        simulated_annealing_params.dist_offset = 50
        simulated_annealing_params.dist_scale = 400
        simulated_annealing_params.temperature_reduction_method = temperature_exp
        simulated_annealing_params.temperature_speed = 0.95

        parameters.launch_count_random_search += 2
        parameters.launch_count_simulated_annealing = 10
    elif tags.equation_type in [EquationType.EXP, EquationType.LIN]:
        parameters.iteration_count += 30

    if tags.model_precision == RequiredModelPrecision.MINIMAL:
        parameters.metric_threshold *= 2
    if tags.model_precision == RequiredModelPrecision.MEDIAN:
        parameters.iteration_count = int(10 * parameters.iteration_count)
    if tags.model_precision == RequiredModelPrecision.MAXIMAL:
        parameters.metric_threshold /= 10
        parameters.iteration_count = int(40 * parameters.iteration_count)
        parameters.max_train_epoch = 700

    if tags.predict_time == ModelPredictTime.SHORT:
        parameters.nn_max_length -= 1
        parameters.nn_min_length -= 1
    elif tags.predict_time == ModelPredictTime.LONG:
        parameters.nn_max_length += 1

    if tags.data_size == DataSize.AUTO:
        if data is None:
            tags.data_size = DataSize.SMALL
        else:
            size = len(data[0])
            size_id = (
                0 + int(size // 100 > 0) + int(size // 300 > 0) + int(size // 900 > 0)
            )
            match size_id:
                case 0:
                    tags.data_size = DataSize.VERY_SMALL
                case 1:
                    tags.data_size = DataSize.SMALL
                case 2:
                    tags.data_size = DataSize.MEDIAN
                case 3:
                    tags.data_size = DataSize.BIG
    if tags.data_size == DataSize.VERY_SMALL:
        parameters.min_train_epoch *= 2
        parameters.max_train_epoch = 700
        parameters.iteration_count += 10
        parameters.launch_count_random_search += 2
        parameters.launch_count_simulated_annealing += 2
    elif tags.data_size == DataSize.SMALL:
        parameters.min_train_epoch = int(parameters.min_train_epoch * 1.5)
        parameters.iteration_count += 10
        parameters.launch_count_random_search += 1
        parameters.launch_count_simulated_annealing += 1
    elif tags.data_size == DataSize.MEDIAN:
        parameters.min_train_epoch = int(parameters.min_train_epoch * 1.25)
        parameters.iteration_count += 10
        parameters.launch_count_random_search += 1
    elif tags.data_size == DataSize.BIG:
        parameters.launch_count_random_search += 1

    parameters.simulated_annealing_params = simulated_annealing_params
    return parameters
