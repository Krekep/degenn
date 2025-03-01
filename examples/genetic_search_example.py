from os import read
from degann.search_algorithms import grid_search, simulated_annealing, random_search
from degann.search_algorithms.genetic_search_iml.genetic_search import genetic_search
import numpy as np
from experiments.functions import gauss
from random import randint
import time
from degann.networks.imodel import *


def genetic_search_example() -> None:
    nn_data_x = np.array([[i / 100] for i in range(0, 1_01)])  # X data
    nn_data_y = np.array([gauss(x) for x in nn_data_x])
    train_idx = [randint(0, len(nn_data_x) - 1) for _ in range(60)]
    train_idx.sort()
    val_idx = [randint(0, len(nn_data_x) - 1) for _ in range(20)]
    val_idx.sort()
    val_data_x = nn_data_x[val_idx, :]  # validation X data
    val_data_y = nn_data_y[val_idx, :]  # validation Y data
    nn_data_x = nn_data_x[train_idx, :]  # X data
    nn_data_y = nn_data_y[train_idx, :]  # Y data
    start_time = time.perf_counter()
    best_loss, best_epoch, best_loss_func, best_opt, best_net = genetic_search(
        input_size=1,
        output_size=1,
        train_data=(nn_data_x, nn_data_y),
        val_data=(
            val_data_x,
            val_data_y,
        ),  # now it should be not null always and not empty
        NNepoch=200,  # epohs for train NN
        neuron_num={i for i in range(1, 10)},  # num of neurons in layer from nn_code
        activ_func={5, 6},  # activation func ind from nn_code
        min_layers=2,
        max_layers=5,
        pop_size=50,
        ngen=20,
        logToConsole=True,
    )
    end_time = time.perf_counter()
    genetic_search_time = end_time - start_time
    print(f"gs result: ", genetic_search_time, best_loss)
