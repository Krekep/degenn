from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from degann.networks import imodel
from degann.networks import utils


def _build_table(
    network: imodel.IModel, axes: List[Tuple[str, Tuple[float, float, float]]], acc=None
) -> List:
    """
    Supporting method for taken network answer.

    Parameters
    ----------
    network: imodel.IModel
        Neural network for build table
    axes: list[tuple[str, tuple[float, float, float]]]
        List of variables with parameters (left, right and step).
    acc: np.ndarray
        Supporting array for variables value
    Returns
    -------
    table: list
        Table with n+k column, where n is the number of variables,
        the first k columns contain the response of the neural network,
        and the last n columns in each row are the values of the variables in the same order as they were input
    """
    if axes:  # if there is non-visited variables
        if acc is None:
            acc = np.array([])
        curr_axis = axes.pop()  # take last variable (left, right, step)
        solution_table = []
        i = curr_axis[1][0]  # left bound
        while i <= curr_axis[1][1]:  # i less than right bound
            tacc = np.append(acc, [i])  # add i to acc
            res = _build_table(network, axes, tacc)  # go to next variable
            for temp in res:
                temp.append(i)
                solution_table.append(temp)
            i += curr_axis[1][2]  # i = i + step
        axes.append(
            curr_axis
        )  # return variable back to axes and go to previous variable
        return solution_table  # return m arrays like [xn, xn-1, ..., x3, x2, x1, y]
    elif acc is not None:
        temp = network.feedforward(
            acc.reshape((1, acc.shape[0]))
        )  # get network answer for X vector
        res = temp.numpy().tolist()  # transform tf.Tensor to python list
        return res
    else:
        raise RuntimeError("This if/else branch shouldn't be reached")


def build_table(
    network: imodel.IModel, axes: List[Tuple[str, Tuple[float, float, float]]]
) -> List:
    """
    Builds a solution table on the interval given for each variable with the given step.

    Parameters
    ----------
    network: imodel.IModel
        Neural network for build table
    axes: list[tuple[str, tuple[float, float, float]]]
        List of variables with parameters (left, right and step).
    Returns
    -------
    table: list
        Table with n+k column, where n is the number of variables,
        the first n columns in each row are the values of the variables in the same order as they were input,
        and the last k columns contain the response of the neural network
    """
    table = _build_table(network, axes)
    res = []  # array of pairs of X vector and Y vector
    k = network.get_output_size
    for i in table:
        res.append([*i[k:], *i[:k]])
    return res


def _equation_solve(
    eq: str, axes: list[tuple[str, tuple[float, float, float]]]
) -> list:
    """
    Supporting method for building a table with solutions of the given equation in points.

    Parameters
    ----------
    eq: str
        Neural network for build table
    axes: list[tuple[str, tuple[float, float, float]]]
        List of variables with parameters (left, right and step).
    Returns
    -------
    table: list
        Table with n+1 column, where n is the number of variables,
        the first 1 column contain the result of the expression,
        and the last n columns in each row are the values of the variables in the same order as they were input
    """
    if axes:  # if there is non-visited variables
        curr_axis = axes.pop()  # take last variable (left, right, step)
        solution_table = []
        i = curr_axis[1][0]  # left bound
        while i <= curr_axis[1][1]:  # i less than right bound
            teq = eq.replace(
                curr_axis[0], "(" + str(i) + ")"
            )  # replace in string repr *var_name* by i
            res = _equation_solve(teq, axes)  # go to next variable
            for temp in res:
                temp.append(i)
                solution_table.append(temp)
            i += curr_axis[1][2]  # i = i + step
        axes.append(
            curr_axis
        )  # return variable back to axes and go to previous variable
        return solution_table  # return m arrays like [xn, xn-1, ..., x3, x2, x1, y]
    else:
        eq_calc = compile(eq, "eq_compile_log.txt", "eval")
        return [[eval(eq_calc)]]


def equation_solve(
    eq: str, axes: list[tuple[str, tuple[float, float, float]]], debug=False
) -> np.ndarray:
    """
    Method for building a table with solutions of the given equation in points.

    Parameters
    ----------
    eq: str
        Neural network for build table
    axes: list[tuple[str, tuple[float, float, float]]]
        List of variables with parameters (left, right and step).
    debug: bool
        Is debug output enabled.
    Returns
    -------
    table: np.ndarray
        Table with n+1 column, where n is the number of variables,
        the first n columns in each row are the values of the variables in the same order as they were input,
        and the last 1 column contain the result of the expression
    """
    table = _equation_solve(eq, axes)
    t = []
    for i in table:
        t.append([*i[1:], i[0]])
    res = np.array(t)
    if debug:
        utils.export_csv_table(res, "solveSimpleEqTable.csv")

        if len(axes) == 1:
            plt.rc("font", size=24)
            plt.figure()
            temp_t = res[:, :1]
            y = res[:, 1:].T

            for i, y_i in enumerate(y):
                plt.plot(temp_t, y_i, "-", label=f"{i}")
            plt.legend()
            plt.show()
    return res


def str_eq_to_params(variables: dict) -> list[tuple[str, tuple[float, float, float]]]:
    """
    Transform string representation of parameters
    to tuple of *var_name* and float parameters (left_bound, right_bound, step)
    for each variable in passed dict

    Parameters
    ----------
    variables: dict
        *var_name* -> string parameters "left_bound, right_bound, step"

    Returns
    -------
        list of tuples of pairs *var_name* and its parameters
    """

    res = []
    for key in variables:
        l, r, s = map(float, variables[key].split(","))
        record = (key, (l, r, s))
        res.append(record)
    return res
