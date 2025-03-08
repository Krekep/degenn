import pytest
import numpy as np
from degann.networks.imodel import IModel
from degann.networks.topology.topology_parameters import TensorflowDenseNetParams
from tests.utils import array_compare, init_params
from degann.equations import simple_equation


@pytest.mark.parametrize(
    "inp, expected",
    [
        (
            {
                "x": "0, 10, 1",
                "y": "-10, 0, 1",
                "z": "0, 5, 1",
            },
            [("x", (0.0, 10.0, 1.0)), ("y", (-10.0, 0.0, 1.0)), ("z", (0.0, 5.0, 1.0))],
        )
    ],
)
def test_str_vars_to_float_vars(inp, expected):
    actual = simple_equation.str_eq_to_params(inp)

    assert array_compare(actual, expected)


@pytest.mark.parametrize(
    "eq, eq_vars, expected",
    [
        (
            "3*x",
            {"x": "0, 4, 1"},
            np.array(
                [
                    [0, 0],
                    [1, 3],
                    [2, 6],
                    [3, 9],
                    [4, 12],
                ],
                dtype=float,
            ),
        ),
        (
            "3*x+2*y+4*z",
            {
                "x": "0, 2, 1",
                "y": "0, 2, 2",
                "z": "0, 0, 1",
            },
            np.array(
                [
                    [0, 0, 0, 0],
                    [1, 0, 0, 3],
                    [2, 0, 0, 6],
                    [0, 2, 0, 4],
                    [1, 2, 0, 7],
                    [2, 2, 0, 10],
                ],
                dtype=float,
            ),
        ),
    ],
)
def test_str_vars_to_float_vars(eq, eq_vars, expected):
    variables = simple_equation.str_eq_to_params(eq_vars)
    actual = simple_equation.equation_solve(eq, variables)

    assert array_compare(actual, expected)


@pytest.mark.parametrize(
    "eq_vars, shape, act_init, w_init, b_init, expected",
    [
        (
            {"x": "0, 4, 1"},
            [1, [], 1],
            "linear",
            "ones",
            "ones",
            np.array(
                [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [4, 5],
                ],
                dtype=float,
            ),
        ),
        (
            {
                "x": "0, 2, 1",
                "y": "0, 2, 2",
                "z": "0, 0, 1",
            },
            [3, [], 1],
            "linear",
            "ones",
            "ones",
            np.array(
                [
                    [0, 0, 0, 1],
                    [1, 0, 0, 2],
                    [2, 0, 0, 3],
                    [0, 2, 0, 3],
                    [1, 2, 0, 4],
                    [2, 2, 0, 5],
                ],
                dtype=float,
            ),
        ),
    ],
)
def test_build_network_answer(eq_vars, shape, act_init, w_init, b_init, expected):
    weight_initializer, bias_initializer = init_params(
        weight_name=w_init, bias_name=b_init
    )

    nn_cfg = TensorflowDenseNetParams(
        input_size=shape[0],
        block_size=shape[1],
        output_size=shape[2],
        activation_func=act_init,
        weight=weight_initializer,
        biases=bias_initializer,
    )
    nn = IModel(nn_cfg)

    variables = simple_equation.str_eq_to_params(eq_vars)
    actual = simple_equation.build_table(nn, variables)

    assert array_compare(actual, expected)
