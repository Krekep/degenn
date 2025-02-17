from datetime import datetime
from itertools import product
from typing import List, Tuple, Optional, Any

from ..nn_code import alph_n_full, alphabet_activations, decode
from degann.networks.callbacks import MeasureTrainTime
from degann.networks import imodel
from ..utils import update_random_generator, log_to_file


from deap import base, creator, tools, algorithms  # type: ignore
import numpy as np
import random
from copy import copy
import traceback


from degann.search_algorithms.grid_search import (
    grid_search_step,
)  # for generating model and validate it


def genetic_topology_repr_encode(genome: List[Tuple[int, int]]) -> str:
    """Encodes a genome (list of layer-activation pairs) into a string representation.

    Args:
        genome (List[Tuple[int, int]]): The genome to encode. Each tuple represents a layer and its activation function.

    Returns:
        str: A string representation of the genome.
    """
    return "".join(f"{layer:x}{activation:x}" for layer, activation in genome)


def genetic_topology_repr_decode(str_genome_repr: str) -> List[Tuple[int, int]]:
    """Decodes a string representation of a genome back into a list of layer-activation pairs.

    Args:
        str_genome_repr (str): The string representation of the genome.

    Returns:
        List[Tuple[int, int]]: The decoded genome as a list of tuples, where each tuple represents a layer and its activation function.
    """
    genome = []
    for i in range(0, len(str_genome_repr), 2):
        layer = int(str_genome_repr[i], 16)
        activation = int(str_genome_repr[i + 1], 16)
        genome.append((layer, activation))
    return genome


def evaluate(
    individual: List[Tuple[int, int]],
    bestNN: List[Any],
    topologyHistory: dict[str, float],
    input_size: int,
    output_size: int,
    train_data: tuple,
    val_data: tuple,
    NNepoch: int,
) -> Tuple[float]:
    """Evaluates an individual (genome) by training a neural network with the specified topology.

    This function checks if the topology has already been evaluated. If not, it trains a neural network,
    records the validation loss, and updates the `bestNN` if the current individual performs better.

    Args:
        individual (List[Tuple[int, int]]): The individual (genome) to evaluate.
        bestNN (List): A list containing the best neural network found so far and its validation loss.
                       `bestNN[0]` is the neural network model, and `bestNN[1]` is the validation loss.
        topologyHistory (dict[str, float]): A dictionary storing the validation loss for each evaluated topology.
                                           Keys are string representations of the topology, and values are the corresponding validation losses.
        input_size (int): The size of the input data.
        output_size (int): The size of the output data.
        train_data (Tuple): The training data.
        val_data (Tuple): The validation data.
        NNepoch (int): The number of epochs to train the neural network.

    Returns:
        Tuple[float]: A tuple containing the best loss.
    """
    topology_str = genetic_topology_repr_encode(individual)

    # Check if topology has already been evaluated
    if topology_str in topologyHistory:
        return (topologyHistory[topology_str],)
    else:
        # Train the NN and get results
        bl, bvl, bn = grid_search_step(
            input_size=input_size,
            output_size=output_size,
            code=topology_str,
            num_epoch=NNepoch,
            opt="Adam",
            loss="MeanSquaredError",
            data=train_data,
            val_data=val_data,
        )

        # Store the validation loss in the topology history
        topologyHistory[topology_str] = bvl

        # Update the best NN if the current one is better
        bestNN_val = bestNN
        if bestNN_val[1] > bvl:
            bestNN_val[:] = [
                bn,
                bvl,
            ]  # if we don't use [:] then it dosen't chenge the passed object bestNN_val
        return (bl,)


def generate_genome(
    min_layers: int, max_layers: int, neuron_num: set[int], activ_func: set[int]
) -> List[Tuple[int, int]]:
    """Generates a random genome (neural network topology).

    Args:
        min_layers (int): The minimum number of layers in the genome.
        max_layers (int): The maximum number of layers in the genome.
        neuron_num (set[int]): A set of possible numbers of neurons for each layer.
        activ_func (set[int]): A set of possible activation function indices for each layer.

    Returns:
        List[Tuple[int, int]]: A list of tuples representing the genome. Each tuple contains the number of neurons and the activation function index for a layer.
    """
    num_layers = random.randint(min_layers, max_layers)
    genome = []
    for _ in range(num_layers):
        genome.append(
            (random.choice(list(neuron_num)), random.choice(list(activ_func)))
        )
    return genome


def mutate_individual(
    individual: List[Tuple[int, int]],
    indpb: float,
    neuron_num: set[int],
    activ_func: set[int],
) -> Tuple[List[Tuple[int, int]]]:
    """Mutates an individual (genome) by randomly changing the number of neurons or activation function in each layer.

    Args:
        individual (List[Tuple[int, int]]): The individual (genome) to mutate.
        indpb (float): The probability of mutating each layer.
        neuron_num (set[int]): A set of possible numbers of neurons for each layer.
        activ_func (set[int]): A set of possible activation function indices for each layer.

    Returns:
        Tuple[List[Tuple[int, int]]]: A tuple containing the mutated individual.
    """
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = (
                random.choice(list(neuron_num)),
                random.choice(list(activ_func)),
            )
    return (individual,)


def mate(
    ind1: List[Tuple[int, int]],
    ind2: List[Tuple[int, int]],
    topologyHistory: dict[str, float],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Mates two individuals (genomes) using one-point crossover.

    Args:
        ind1 (List[Tuple[int, int]]): The first individual.
        ind2 (List[Tuple[int, int]]): The second individual.
        topologyHistory (dict[str, float]): A dictionary storing the validation loss for each evaluated topology.

    Returns:
        Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]: A tuple containing the two offspring.
    """
    for _ in range(10):
        ind1_pr = copy(ind1)
        ind2_pr = copy(ind2)
        ind1_pr, ind2_pr = tools.cxOnePoint(ind1_pr, ind2_pr)
        if (
            genetic_topology_repr_encode(ind1_pr) not in topologyHistory
            and genetic_topology_repr_encode(ind2_pr) not in topologyHistory
        ):
            break

    ind1 = ind1_pr
    ind2 = ind2_pr
    return ind1, ind2


def genetic_search(
    # NN train haracteristics
    input_size: int = 1,
    output_size: int = 1,
    train_data: Optional[tuple] = None,
    val_data: Optional[tuple] = None,  # now it should be not null always and not empty
    NNepoch: int = 10,  # epohs for train NN
    # topology haracteristics
    neuron_num: set[int] = {1, 2, 3, 4},  # num of neurons in layer from nn_code
    activ_func: set[int] = {1, 2, 3},  # activation func ind from nn_code
    min_layers: int = 2,
    max_layers: int = 4,
    pop_size: int = 10,  # size of start population
    ngen: int = 10,  # how epochs for GA
    # special parametrs like logging or callbacks
    logToConsole: bool = False,
) -> Tuple[float, float, str, str, dict]:
    """Performs a genetic search to find the best neural network topology.

    This function uses a genetic algorithm to explore different neural network topologies
    and find the one that performs best on the given training and validation data.

    Args:
        input_size (int): The size of the input data.
        output_size (int): The size of the output data.
        train_data (Union[tuple, None]): The training data.
        val_data (Union[tuple, None]): The validation data.
        NNepoch (int): The number of epochs to train each neural network during evaluation.
        neuron_num (set[int]): A set of possible numbers of neurons for each layer.
        activ_func (set[int]): A set of possible activation function indices for each layer.
        min_layers (int): The minimum number of layers in the neural network.
        max_layers (int): The maximum number of layers in the neural network.
        pop_size (int): The size of the population in the genetic algorithm.
        ngen (int): The number of generations to run the genetic algorithm for.
        logToConsole (bool): Whether to print the log to the console.

    Returns:
        Tuple[float, float, str, str, dict]: A tuple containing:
            - The validation loss of the best neural network.
            - -1. (Reason unclear from the code, but it's a constant value being returned).
            - "MeanSquaredError" (The loss function used).
            - "Adam" (The optimizer used).
            - The best neural network model (a dictionary).
    """
    bestNN: List = [
        {},
        float("inf"),
    ]  # contain here the best NN evere builded here and it's
    # it is list becouse it should be chengeble when we call
    topologyHistory: dict[str, float] = {}  # contain every topology

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register(
        "generate_genome_with_defoult",  # set the func which generate to us the genome
        generate_genome,
        min_layers=min_layers,
        max_layers=max_layers,
        neuron_num=neuron_num,
        activ_func=activ_func,
    )

    toolbox.register(
        "individual",
        tools.initIterate,
        creator.Individual,
        toolbox.generate_genome_with_defoult,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate",
        evaluate,
        bestNN=bestNN,
        topologyHistory=topologyHistory,
        input_size=input_size,
        output_size=input_size,
        train_data=train_data,
        val_data=val_data,
        NNepoch=NNepoch,
    )
    toolbox.register(
        "mate", mate, topologyHistory=topologyHistory
    )  # maybe it's bad war to our topology finder
    toolbox.register(
        "mutate",
        mutate_individual,
        indpb=0.01,
        neuron_num=neuron_num,
        activ_func=activ_func,
    )  # indpb=0.01
    toolbox.register("select", tools.selTournament, tournsize=3)

    # for understand(see the evolution)
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # evolution we are need for
    try:
        pop, log = algorithms.eaSimple(
            pop,
            toolbox,
            cxpb=0.5,
            mutpb=0.2,
            ngen=ngen,
            stats=stats,
            halloffame=hof,
            verbose=logToConsole,
        )

        best_individual = hof[0]
        print("str repr of the best", genetic_topology_repr_encode(best_individual))
        print(":", best_individual.fitness.values[0])
        print("", sum([layer[0] for layer in best_individual]))

        # now we do not find "opt" and "loss" just use these ones
        return bestNN[1], -1, "MeanSquaredError", "Adam", bestNN[0]

    except Exception as e:  # it's for debug of deap
        print(f"Error in evaluate: {e}")
        traceback.print_exc()
        raise Exception
