from datetime import datetime
from typing import Optional, Tuple

import numpy as np

from degann.networks.callbacks import MeasureTrainTime
from degann.networks import imodel
from degann.networks.topology.utils import generate_all_configurations
from .search_algorithms_parameters import GridSearchParameters
from .utils import update_random_generator, log_to_file, SearchHistory, log_search_step


def grid_search_step(
    model_cfg: imodel.BaseTopologyParams,
    compile_cfg: imodel.BaseCompileParams,
    num_epoch: int,
    data: tuple[np.ndarray, np.ndarray],
    repeat: int = 1,
    val_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    update_gen_cycle: int = 0,
    logging: bool = False,
    file_name: str = "",
    callbacks: Optional[list] = None,
    eval_metric: str = "root_mean_squared_error",
):
    """
    This function is a step of the exhaustive search algorithm.
    In this function, the passed neural network is trained (possibly several times).

    Parameters
    ----------
    model_cfg: BaseTopologyParams
        Config for model to train
    compile_cfg: BaseCompileParams
        Config for model compilation
    num_epoch: int
        Number of training epochs
    data: tuple
        Dataset
    repeat: int
        How many times will be repeated this step
    val_data: Optional[tuple[np.ndarray, np.ndarray]]
        Validation dataset
    logging: bool
        Logging search process to file
    file_name: str
        Path to file for logging
    eval_metric: str
        Metric used for model evaluation

    Returns
    -------
    search_results: tuple[float, int, str, str, dict]
        Results of the algorithm are described by these parameters

        best_metric_value: float
            The value of the metric during training of the best neural network]
        best_val_metric_value: Optional[float]
            The corresponding validation metric value (if validation data is provided).
        best_net: dict
            Best neural network presented as a dictionary
    """
    best_net = None
    best_metric_value = 1e6
    best_val_metric_value: Optional[float] = 1e6
    for i in range(repeat):
        update_random_generator(i, cycle_size=update_gen_cycle)
        history = SearchHistory()

        nn = imodel.IModel(model_cfg)
        compile_cfg.add_eval_metric(eval_metric)
        nn.compile(compile_cfg)

        temp_his = nn.train(
            data[0], data[1], epochs=num_epoch, verbose=0, callbacks=callbacks
        )

        loss_names = nn.network.get_loss_names()
        curr_loss = [temp_his.history[name][-1] for name in loss_names]
        curr_metric_value = temp_his.history[eval_metric][-1]

        if val_data is not None:
            val_history = nn.evaluate(
                val_data[0], val_data[1], verbose=0, return_dict=True
            )
            eval_loss = [val_history[name] for name in loss_names]
            val_metric_value = val_history[eval_metric]
        else:
            eval_loss = None
            val_metric_value = None

        if logging:
            str_losses = "_".join(compile_cfg.get_losses())
            str_optimizers = "_".join(compile_cfg.get_optimizers())
            fn = f"{file_name}_{len(data[0])}_{num_epoch}_{str_losses}_{str_optimizers}"
            log_search_step(
                model=nn,
                activations=nn.get_activations(),
                epoch=num_epoch,
                optimizers=str_optimizers,
                loss_functions=str_losses,
                loss=curr_loss,
                validation_loss=eval_loss,
                metric_value=curr_metric_value,
                validation_metric_value=val_metric_value,
                file_name=fn,
            )
            log_to_file(history.__dict__, fn)

        if curr_metric_value < best_metric_value:
            best_metric_value = curr_metric_value
            best_val_metric_value = val_metric_value
            best_net = nn.to_dict()

    return (best_metric_value, best_val_metric_value, best_net)


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
    best_net: dict = dict()
    best_metric_value: float = 1e6
    best_epoch: int = 0
    best_loss_func: str = ""
    best_opt: str = ""
    time_viewer = MeasureTrainTime()

    for model_cfg in generate_all_configurations(parameters.model_cfg):
        for compile_cfg in generate_all_configurations(parameters.compile_cfg):
            for epoch in range(
                parameters.min_epoch, parameters.max_epoch + 1, parameters.epoch_step
            ):
                (
                    curr_metric_value,
                    curr_val_metric_value,
                    curr_nn,
                ) = grid_search_step(
                    model_cfg=model_cfg,
                    compile_cfg=compile_cfg,
                    num_epoch=epoch,
                    data=parameters.data,
                    val_data=parameters.val_data,
                    callbacks=[time_viewer],
                    logging=parameters.logging,
                    file_name=parameters.file_name,
                    eval_metric=parameters.eval_metric,
                )
                if best_metric_value > curr_metric_value:
                    best_net = curr_nn
                    best_metric_value = curr_metric_value
                    best_epoch = epoch
                    best_loss_func = compile_cfg.get_losses()
                    best_opt = compile_cfg.get_optimizers()
    return best_metric_value, best_epoch, best_loss_func, best_opt, best_net
