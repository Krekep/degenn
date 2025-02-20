from datetime import datetime
from itertools import product
from typing import Optional, List, Tuple

import numpy as np

from .nn_code import decode, default_alphabet
from degann.networks.callbacks import MeasureTrainTime
from degann.networks import imodel
from .search_algorithms_parameters import GridSearchParameters
from .utils import update_random_generator, log_to_file, SearchHistory, log_search_step


def grid_search_step(
    input_size: int,
    output_size: int,
    code: str,
    num_epoch: int,
    opt: str,
    loss: str,
    data: tuple,
    repeat: int = 1,
    alphabet_block_size: int = 1,
    alphabet_offset: int = 8,
    val_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    update_gen_cycle: int = 0,
    logging: bool = False,
    file_name: str = "",
    callbacks: Optional[list] = None,
    metrics: Optional[list[str]] = None,
):
    """
    This function is a step of the exhaustive search algorithm.
    In this function, the passed neural network is trained (possibly several times).

    Parameters
    ----------
    input_size: int
       Size of input data
    output_size: int
        Size of output data
    code: str
        Neural network as code
    num_epoch: int
        Number of training epochs
    data: tuple
        Dataset
    opt: str
        Optimizer
    loss: str
        Name of loss function
    repeat: int
        How many times will be repeated this step
    alphabet_block_size: int
        Number of literals in each `alphabet` symbol that indicate the size of hidden layer
    alphabet_offset: int
        Indicate the minimal number of neurons in hidden layer
    val_data: Optional[tuple[np.ndarray, np.ndarray]]
        Validation dataset
    logging: bool
        Logging search process to file
    file_name: str
        Path to file for logging
    metrics: Optional[list[str]]
        List of metrics for model

    Returns
    -------
    search_results: tuple[float, int, str, str, dict]
        Results of the algorithm are described by these parameters

        best_loss: float
            The value of the loss function during training of the best neural network
        best_epoch: int
            Number of training epochs for the best neural network
        best_loss_func: str
            Name of the loss function of the best neural network
        best_opt: str
            Name of the optimizer of the best neural network
        best_net: dict
            Best neural network presented as a dictionary
    """
    best_net = None
    best_loss = 1e6
    best_val_loss: Optional[float] = 1e6
    for i in range(repeat):
        update_random_generator(i, cycle_size=update_gen_cycle)
        history = SearchHistory()
        b, a = decode(code, block_size=alphabet_block_size, offset=alphabet_offset)
        nn = imodel.IModel(input_size, b, output_size, a + ["linear"])
        nn.compile(optimizer=opt, loss_func=loss, metrics=metrics)
        temp_his = nn.train(
            data[0], data[1], epochs=num_epoch, verbose=0, callbacks=callbacks
        )

        curr_loss = temp_his.history["loss"][-1]
        if val_data is not None:
            eval_loss = nn.evaluate(
                val_data[0], val_data[1], verbose=0, return_dict=True
            )["loss"]
        else:
            eval_loss = None

        if logging:
            fn = f"{file_name}_{len(data[0])}_{num_epoch}_{loss}_{opt}"
            log_search_step(
                model=nn,
                activations=a,
                code=code,
                epoch=num_epoch,
                optimizer=opt,
                loss_function=loss,
                loss=curr_loss,
                validation_loss=eval_loss,
                file_name=fn,
            )

        if logging:
            fn = f"{file_name}_{len(data[0])}_{num_epoch}_{loss}_{opt}"
            log_to_file(history.__dict__, fn)
        if curr_loss < best_loss:
            best_loss = curr_loss
            best_val_loss = eval_loss
            best_net = nn.to_dict()
    return (best_loss, best_val_loss, best_net)


def grid_search(
    parameters: GridSearchParameters,
    verbose: bool = False,
) -> Tuple[float, int, str, str, dict]:
    """
    An algorithm for exhaustively enumerating a given set of parameters
    with training a neural network for each configuration of parameters
    and selecting the best one.

    Parameters
    ----------
    parameters: GridSearchParameters
        Search algorithm parameters
    verbose: bool
        Print additional information to console during the searching

    Returns
    -------
    search_results: tuple[float, int, str, str, dict]
        Results of the algorithm are described by these parameters

        best_loss: float
            The value of the loss function during training of the best neural network
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

    best_net: dict = dict()
    best_loss: float = 1e6
    best_epoch: int = 0
    best_loss_func: str = ""
    best_opt: str = ""
    time_viewer = MeasureTrainTime()
    for i in range(parameters.nn_min_length, parameters.nn_max_length + 1):
        if verbose:
            print(i, datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
        codes = product(parameters.nn_alphabet, repeat=i)
        for elem in codes:
            code = "".join(elem)
            for epoch in range(
                parameters.min_epoch, parameters.max_epoch + 1, parameters.epoch_step
            ):
                for opt in parameters.optimizers:
                    for loss_func in parameters.losses:
                        curr_loss, curr_val_loss, curr_nn = grid_search_step(
                            input_size=parameters.input_size,
                            output_size=parameters.output_size,
                            code=code,
                            num_epoch=epoch,
                            opt=opt,
                            loss=loss_func,
                            data=parameters.data,
                            alphabet_block_size=parameters.nn_alphabet_block_size,
                            alphabet_offset=parameters.nn_alphabet_offset,
                            val_data=parameters.val_data,
                            callbacks=[time_viewer],
                            logging=parameters.logging,
                            file_name=parameters.file_name,
                            metrics=parameters.metrics,
                        )
                        if best_loss > curr_loss:
                            best_net = curr_nn
                            best_loss = curr_loss
                            best_epoch = epoch
                            best_loss_func = loss_func
                            best_opt = opt
    return best_loss, best_epoch, best_loss_func, best_opt, best_net
