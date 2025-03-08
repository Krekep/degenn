import pytest

import numpy as np
from degann.search_algorithms import (
    pattern_search,
    grid_search,
    random_search_endless,
    simulated_annealing,
)
from degann.search_algorithms.search_algorithms_parameters import (
    BaseSearchParameters,
    GridSearchParameters,
    RandomSearchParameters,
    RandomEarlyStoppingSearchParameters,
    SimulatedAnnealingSearchParameters,
)
from degann.search_algorithms.simulated_annealing_functions import distance_lin

from degann.networks.topology.topology_parameters import (
    TensorflowDenseNetParams,
    GANTopologyParams,
)
from degann.networks.topology.compile_parameters import (
    SingleNetworkCompileParams,
    GANCompileParams,
)
from degann.networks.topology.utils import TuningMetadata


@pytest.fixture
def train_file_name():
    return "exp_150_train.csv"


@pytest.fixture
def validate_file_name():
    return "exp_150_validate.csv"


@pytest.fixture
def equation_data(train_file_name, validate_file_name):
    folder_path = "./tests/data"
    train_data = np.genfromtxt(folder_path + "/" + train_file_name, delimiter=",")
    train_data_x, train_data_y = train_data[:, 0], train_data[:, 1]
    train_data_x = train_data_x.reshape((1, -1)).T
    train_data_y = train_data_y.reshape((1, -1)).T

    validation_data = np.genfromtxt(
        folder_path + "/" + validate_file_name, delimiter=","
    )
    validation_data_x, validation_data_y = validation_data[:, 0], validation_data[:, 1]
    validation_data_x = validation_data_x.reshape((1, -1)).T
    validation_data_y = validation_data_y.reshape((1, -1)).T

    return ((train_data_x, train_data_y), (validation_data_x, validation_data_y))


def test_pattern_search(equation_data):
    train_data_x = equation_data[0][0]
    train_data_y = equation_data[0][1]

    validation_data_x = equation_data[1][0]
    validation_data_y = equation_data[1][1]

    config = {
        "loss_functions": ["MeanSquaredError"],
        "optimizers": ["Adam"],
        "eval_metric": "root_mean_squared_error",
        "net_shapes": [[10], [5]],
        "activations": ["parabolic"],
        "validation_split": 0,
        "rates": [1e-2],
        "epochs": [5],
        "normalize": [False],
        "use_rand_net": False,
    }

    best_nns = pattern_search(
        x_data=train_data_x,
        y_data=train_data_y,
        x_val=validation_data_x,
        y_val=validation_data_y,
        **config
    )
    assert True


@pytest.mark.parametrize(
    "in_size, out_size",
    [
        (
            1,
            1,
        ),
    ],
)
def test_grid_search(equation_data, in_size, out_size):
    train_data_x = equation_data[0][0]
    train_data_y = equation_data[0][1]

    validation_data_x = equation_data[1][0]
    validation_data_y = equation_data[1][1]

    model_metadata = {
        "block_size": TuningMetadata(value_range=(10, 20, 10), length_boundary=(1, 1)),
        "activation_func": TuningMetadata(choices=["sigmoid", "relu"]),
    }
    model_cfg = TensorflowDenseNetParams(
        input_size=in_size, output_size=out_size, tuning_metadata=model_metadata
    )

    compile_metadata = {
        "optimizer": TuningMetadata(choices=["SGD", "Adam"]),
    }
    compile_cfg = SingleNetworkCompileParams(tuning_metadata=compile_metadata)

    search_alg_params = BaseSearchParameters()
    search_alg_params.model_cfg = model_cfg
    search_alg_params.compile_cfg = compile_cfg
    search_alg_params.data = (train_data_x, train_data_y)
    search_alg_params.val_data = (validation_data_x, validation_data_y)

    grid_search_parameters = GridSearchParameters(search_alg_params)
    grid_search_parameters.min_epoch = 5
    grid_search_parameters.max_epoch = 10
    grid_search_parameters.epoch_step = 5

    (
        result_metric_value,
        result_epoch,
        result_loss_name,
        result_optimizer,
        result_nn,
    ) = grid_search(grid_search_parameters)

    # Grid search for GAN

    # Search the number of neurons in a layer (10 or 20) and the number of layers (1 or 2)
    generator_metadata = {
        "block_size": TuningMetadata(
            value_range=(10, 20, 10),
        ),
    }
    generator_cfg = TensorflowDenseNetParams(
        tuning_metadata=generator_metadata, activation_func="relu"
    )

    # Search the number of neurons in a layer (10 or 20)
    discriminator_metadata = {
        "block_size": TuningMetadata(value_range=(10, 20, 10), length_boundary=(1, 2)),
    }
    discriminator_cfg = TensorflowDenseNetParams(
        tuning_metadata=discriminator_metadata, input_size=2, activation_func="relu"
    )

    GAN_config = GANTopologyParams(
        generator_params=generator_cfg, discriminator_params=discriminator_cfg
    )

    generator_compile_config = SingleNetworkCompileParams(
        optimizer="Adam", loss_func="BinaryCrossentropy", metric_funcs=[]
    )
    # Run with 2 optimisers: Adam and SGD
    compile_metadata = {
        "optimizer": TuningMetadata(choices=["SGD", "Adam"]),
    }
    discriminator_compile_config = SingleNetworkCompileParams(
        tuning_metadata=compile_metadata,
        loss_func="BinaryCrossentropy",
        metric_funcs=[],
    )
    GAN_compile_cfg = GANCompileParams(
        generator_params=generator_compile_config,
        discriminator_params=discriminator_compile_config,
    )

    search_alg_params = BaseSearchParameters()
    search_alg_params.model_cfg = GAN_config
    search_alg_params.compile_cfg = GAN_compile_cfg
    search_alg_params.data = (train_data_x, train_data_y)
    search_alg_params.val_data = (validation_data_x, validation_data_y)

    grid_search_parameters = GridSearchParameters(search_alg_params)
    grid_search_parameters.min_epoch = 5
    grid_search_parameters.max_epoch = 10
    grid_search_parameters.epoch_step = 5

    (
        result_metric_value,
        result_epoch,
        result_loss_name,
        result_optimizer,
        result_nn,
    ) = grid_search(grid_search_parameters)

    assert True


@pytest.mark.parametrize(
    "in_size, out_size",
    [
        (
            1,
            1,
        ),
    ],
)
def test_random_search(equation_data, in_size, out_size):
    train_data_x = equation_data[0][0]
    train_data_y = equation_data[0][1]

    validation_data_x = equation_data[1][0]
    validation_data_y = equation_data[1][1]

    search_alg_params = BaseSearchParameters()
    search_alg_params.input_size = in_size
    search_alg_params.output_size = out_size
    search_alg_params.data = (train_data_x, train_data_y)
    search_alg_params.val_data = (validation_data_x, validation_data_y)

    random_search_parameters = RandomEarlyStoppingSearchParameters(search_alg_params)
    random_search_parameters.optimizer = "Adam"
    random_search_parameters.loss_function = "MaxAbsolutePercentageError"
    random_search_parameters.min_epoch = 5
    random_search_parameters.max_epoch = 10
    random_search_parameters.metric_threshold = 2.5
    random_search_parameters.nn_min_length = 1
    random_search_parameters.nn_max_length = 3
    random_search_parameters.nn_alphabet = ["0a", "f8", "42"]
    random_search_parameters.iterations = 1
    random_search_parameters.max_launches = -1

    (
        result_metric_value,
        result_epoch,
        result_loss_name,
        result_optimizer,
        result_nn,
        final_iteration,
    ) = random_search_endless(random_search_parameters)
    assert result_metric_value < random_search_parameters.metric_threshold


@pytest.mark.parametrize(
    "in_size, out_size",
    [
        (
            1,
            1,
        ),
    ],
)
def test_sam(equation_data, in_size, out_size):
    train_data_x = equation_data[0][0]
    train_data_y = equation_data[0][1]

    validation_data_x = equation_data[1][0]
    validation_data_y = equation_data[1][1]

    search_alg_params = BaseSearchParameters()
    search_alg_params.input_size = in_size
    search_alg_params.output_size = out_size
    search_alg_params.data = (train_data_x, train_data_y)
    search_alg_params.val_data = (validation_data_x, validation_data_y)

    simulated_annealing_parameters = SimulatedAnnealingSearchParameters(
        search_alg_params
    )
    simulated_annealing_parameters.optimizer = "Adam"
    simulated_annealing_parameters.loss_function = "Huber"
    simulated_annealing_parameters.min_epoch = 5
    simulated_annealing_parameters.max_epoch = 10
    simulated_annealing_parameters.metric_threshold = 2.5
    simulated_annealing_parameters.nn_min_length = 1
    simulated_annealing_parameters.nn_max_length = 3
    simulated_annealing_parameters.nn_alphabet = ["0a", "f8", "42"]
    simulated_annealing_parameters.max_launches = 30
    simulated_annealing_parameters.distance_method = distance_lin(400, 50)

    (
        result_metric_value,
        result_epoch,
        result_loss_name,
        result_optimizer,
        result_nn,
        final_iteration,
    ) = simulated_annealing(simulated_annealing_parameters)
    assert result_metric_value < simulated_annealing_parameters.metric_threshold
