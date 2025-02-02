import pytest

import numpy as np
from degann.networks.imodel import IModel
from degann.networks.topology.topology_parameters import TensorflowDenseNetParams
from degann.networks.topology.compile_parameters import SingleNetworkCompileParams


@pytest.fixture
def folder_path():
    return "./tests/data"


@pytest.mark.parametrize(
    "path_to_train_data, path_to_validate_data, shape, out_size",
    [
        (
            "/exp_150_train.csv",
            "/exp_150_validate.csv",
            [1, [10, 10]],
            1,
        ),
    ],
)
def test_densenet_predict(
    path_to_train_data, path_to_validate_data, folder_path, shape, out_size
):
    train_data = np.genfromtxt(folder_path + path_to_train_data, delimiter=",")
    train_data_x, train_data_y = train_data[:, 0], train_data[:, 1]
    train_data_x = train_data_x.reshape((1, -1)).T
    train_data_y = train_data_y.reshape((1, -1)).T

    validation_data = np.genfromtxt(folder_path + path_to_validate_data, delimiter=",")
    validation_data_x, validation_data_y = validation_data[:, 0], validation_data[:, 1]
    validation_data_x = validation_data_x.reshape((1, -1)).T
    validation_data_y = validation_data_y.reshape((1, -1)).T

    nn_cfg = TensorflowDenseNetParams(
        input_size=shape[0],
        block_size=shape[1],
        output_size=out_size,
    )
    nn = IModel(nn_cfg)

    compile_cfg = SingleNetworkCompileParams(
        optimizer="Adam", loss_func="MaxAbsoluteDeviation"
    )
    nn.compile(compile_cfg)

    loss_before_train = nn.evaluate(validation_data_x, validation_data_y, verbose=0)
    nn.train(train_data_x, train_data_y, verbose=0)
    loss_after_train = nn.evaluate(validation_data_x, validation_data_y, verbose=0)
    assert loss_after_train < loss_before_train
