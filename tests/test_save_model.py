import pytest
import numpy as np

from degann.networks.imodel import IModel
from degann.networks.topology.topology_parameters import TensorflowDenseNetParams
from tests.utils import array_compare, file_compare


@pytest.fixture
def folder_path():
    return "./tests/data"


@pytest.mark.parametrize(
    "inp, shape, act_init, decorator_params",
    [
        (np.array([[1]], dtype=float), [1, [1], 1], ["sigmoid", "linear"], None),
        (np.array([[1]], dtype=float), [1, [1], 1], "sigmoid", None),
        (np.array([[1]], dtype=float), [1, [1], 1], ["linear", "linear"], None),
        (np.array([[1, 1]], dtype=float), [2, [1], 1], "tanh", None),
        (np.array([[1], [1]], dtype=float), [1, [1], 1], "tanh", None),
    ],
)
def test_predict_is_same(inp, shape, act_init, decorator_params, folder_path):
    nn_cfg = TensorflowDenseNetParams(
        input_size=shape[0],
        block_size=shape[1],
        output_size=shape[2],
        activation_func=act_init,
    )

    nn = IModel(
        nn_cfg,
        decorator_params=decorator_params,
    )

    expected = nn.feedforward(inp).numpy()
    nn.export_to_file(f"{folder_path}/test_export")

    nn_loaded_cfg = TensorflowDenseNetParams(
        input_size=shape[0],
        block_size=shape[1],
        output_size=shape[2],
    )

    nn_loaded = IModel(nn_loaded_cfg)
    nn_loaded.from_file(f"{folder_path}/test_export")
    nn_loaded.export_to_file(f"{folder_path}/test_export1")
    actual = nn_loaded.feedforward(inp).numpy()

    assert array_compare(actual, expected)


@pytest.mark.parametrize(
    "inp, shape",
    [
        (
            np.array([[1]], dtype=float),
            [1, [1], 1],
        ),
        (
            np.array([[1]], dtype=float),
            [1, [1], 1],
        ),
        (
            np.array([[1]], dtype=float),
            [1, [1], 1],
        ),
        (
            np.array([[1, 1]], dtype=float),
            [2, [1], 1],
        ),
        (
            np.array([[1], [1]], dtype=float),
            [1, [1], 1],
        ),
        (
            np.array([[1, 1], [1, 1]], dtype=float),
            [2, [1], 2],
        ),
        (
            np.array([[1, 1], [1, 1]], dtype=float),
            [2, [1], 1],
        ),
    ],
)
def test_file_is_same(inp, shape, folder_path):
    cfg = TensorflowDenseNetParams(
        input_size=shape[0],
        block_size=shape[1],
        output_size=shape[2],
    )

    nn = IModel(cfg)
    nn.export_to_file(f"{folder_path}/test_export")

    nn_loaded = IModel(cfg)
    nn_loaded.from_file(f"{folder_path}/test_export")
    nn_loaded.export_to_file(f"{folder_path}/test_export1")

    assert file_compare(
        f"{folder_path}/test_export.apg", f"{folder_path}/test_export1.apg"
    )
