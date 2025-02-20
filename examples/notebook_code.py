from degann.expert import ExpertSystemTags
from degann.expert.tags import (
    EquationType,
    RequiredModelPrecision,
    ModelPredictTime,
    DataSize,
)
from degann.networks.imodel import IModel
from degann.search_algorithms import (
    random_search_endless,
    simulated_annealing,
    pattern_search,
    grid_search,
)
from degann.search_algorithms.nn_code import (
    encode,
    decode,
)

from degann.expert.selector import suggest_parameters
from degann.expert.pipeline import execute_pipeline

from degann.equations import (
    build_plot,
    equation_solve,
    SystemODE,
    system_ode_from_string,
    str_eq_to_params,
)

import numpy as np

from random import randint

from degann.search_algorithms.search_algorithms_parameters import (
    BaseSearchParameters,
    GridSearchParameters,
    RandomEarlyStoppingSearchParameters,
    SimulatedAnnealingSearchParameters,
)


# Prepare data for neural network training


def f_3x2(x):
    return 3 * x**2


data_size = 1000
x_data = np.array([[i / data_size] for i in range(0, data_size + 1)])
y_data = np.array([f_3x2(x) for x in x_data])

train_data_size = 200

train_idx = [randint(0, len(x_data) - 1) for _ in range(train_data_size)]
train_idx.sort()
train_data_x = x_data[train_idx, :]  # X data
train_data_y = y_data[train_idx, :]  # Y data

nn_1_32_16_8_1 = IModel(input_size=1, block_size=[32, 16, 8], output_size=1)
print(nn_1_32_16_8_1)

shape = [10, 10]
activations = ["swish", "relu"] + [
    "linear"
]  # additional activation function for output layer

nn_1_10_10_3 = IModel(
    input_size=1,
    block_size=shape,
    output_size=1,
    activation_func=activations,
)

print("Activation functions per layer for n_1_10_10_3")
acts = nn_1_10_10_3.get_activations
for i, act_name in enumerate(activations):
    print(i, act_name)

print(nn_1_10_10_3)

# Prepare network for training

nn_1_32_16_8_1.compile(
    optimizer="Adam",
    loss_func="MaxAbsoluteDeviation",  # max(abs(y_true - y_prediction))
    metrics=[],
)

# Train network (something about 5 sec. on Google Colab)

loss_before_train = nn_1_32_16_8_1.evaluate(x_data, y_data, verbose=0)

nn_1_32_16_8_1.train(train_data_x, train_data_y, epochs=50, verbose=0)

loss_after_train = nn_1_32_16_8_1.evaluate(x_data, y_data, verbose=0)

print(f"Loss before training = {loss_before_train}")
print(f"Loss after training = {loss_after_train}")

nn_1_32_16_8_1.export_to_file("some_path")
nn_1_32_16_8_1.export_to_cpp("some_path")


# Takes about 2 min. in Google Colab

config = {
    "loss_functions": ["MeanSquaredError"],
    "optimizers": ["Adam"],
    "metrics": ["MaxAbsoluteDeviation", "MeanSquaredLogarithmicError"],
    "net_shapes": [[], [10], [5, 5]],  # neural network without hidden layers
    "activations": ["parabolic", "exponential"],
    "validation_split": 0,
    "rates": [1e-2],
    "epochs": [10],
    "normalize": [False],
    "use_rand_net": False,
}

best_nns = pattern_search(
    x_data=train_data_x, y_data=train_data_y, x_val=x_data, y_val=y_data, **config
)
print(best_nns)

# Examples of coding

shape_1 = [10, 8, 23, 16]
activations_1 = ["tanh", "exponential", "relu", "swish", "linear"]
nn_for_code_1 = IModel(
    input_size=1, block_size=shape_1, output_size=1, activation_func=activations_1
)
code_1 = encode(nn_for_code_1, offset=8)
print(
    f"Neural network with shape {shape_1} and activations {activations_1} encoded in {code_1}"
)

shape_1_from_code, activations_1_from_code = decode(code_1, block_size=1, offset=8)
print(
    f"{code_1} decoded in shape {shape_1_from_code} and activations {activations_1_from_code}"
)

code_2 = "e4aa00"
shape_2_from_code, activations_2_from_code = decode(code_2, block_size=2, offset=1)
print(
    f"{code_2} with block_size=2 and offset=1 decoded in shape {shape_2_from_code} and activations {activations_2_from_code}"
)


base_params = BaseSearchParameters()
base_params.input_size = 1  # size of input data (x)
base_params.output_size = 1  # size of output data (y)
base_params.data = (train_data_x, train_data_y)  # dataset
base_params.min_epoch = 10  # starting number of epochs
base_params.max_epoch = 20  # final number of epochs
base_params.nn_min_length = 1  # starting number of hidden layers of neural networks
base_params.nn_max_length = 2  # final number of hidden layers of neural networks
base_params.nn_alphabet = [
    "0a",
    "42",
]  # list of possible sizes of hidden layers with activations for them
base_params.logging = False  # logging search process to file
base_params.optimizer = "Adam"  # Optimizer

grid_search_params = GridSearchParameters(base_params)
grid_search_params.optimizers = ["Adam"]  # list of optimizers
grid_search_params.losses = ["MeanAbsolutePercentageError"]  # list of loss functions
grid_search_params.epoch_step = 10  # step between `min_epoch` and `max_epoch`


result_loss, result_epoch, result_loss_name, result_optimizer, result_nn = grid_search(
    grid_search_params
)
print(result_nn)

random_search_params = RandomEarlyStoppingSearchParameters(base_params)
random_search_params.nn_max_length = 3
random_search_params.nn_alphabet = ["0a", "f8", "42"]
random_search_params.max_launches = 10
random_search_params.loss_threshold = 20
random_search_params.iterations = 1
random_search_params.loss_function = "MaxAbsolutePercentageError"

(
    result_loss,
    result_epoch,
    result_loss_name,
    result_optimizer,
    result_nn,
    final_iteration,
) = random_search_endless(random_search_params)
print(result_nn)

sim_ann_search_params = SimulatedAnnealingSearchParameters(base_params)
sim_ann_search_params.loss_function = "Huber"
sim_ann_search_params.max_launches = 10
sim_ann_search_params.nn_max_length = 3
sim_ann_search_params.loss_threshold = 1
sim_ann_search_params.nn_alphabet = ["0a", "f8", "42"]

(
    result_loss,
    result_epoch,
    result_loss_name,
    result_optimizer,
    result_nn,
    final_iteration,
) = simulated_annealing(sim_ann_search_params)
print(result_nn)

selector_tags = ExpertSystemTags()
selector_tags.equation_type = (
    EquationType.EXP
)  # type of function in data --- in this case it is the parabola (3 * x^2)
selector_tags.model_precision = (
    RequiredModelPrecision.MINIMAL
)  # Shows how important the accuracy of the solution is to us
selector_tags.predict_time = (
    ModelPredictTime.MEDIUM
)  # Shows how important the operating time (predict) of the resulting neural network is to us
selector_tags.data_size = DataSize.MEDIAN  # Training dataset size
algorithms_parameters = suggest_parameters(tags=selector_tags)

print("Resulting parameters by expert system for search algorithms")
for k, v in algorithms_parameters.__dict__.items():
    print(f"{k}: {v}")

# All possible tags:

# print(expert_system_tags)

# Takes about 30 sec. in Google colab

result_loss, result_nn = execute_pipeline(
    input_size=1,
    output_size=1,
    data=(train_data_x, train_data_y),
    parameters=algorithms_parameters,
)
print("Resulting loss value =", result_loss)
print("Resulting network:")
print(result_nn)

model_from_expert_system = IModel(1, [], 1)
model_from_expert_system.from_dict(result_nn)  # restore model from dict

indices = []
for _ in range(30):
    indices.append(randint(0, len(x_data) - 1))
indices.sort()
x_plot_data = x_data[indices, :]
y_plot_data = y_data[indices, :]

build_plot(
    network=[
        model_from_expert_system,
        nn_1_32_16_8_1,
    ],  # list of models (or single model)
    interval=(0.0, 1.0),  # from where to where to build a plot
    step=0.02,  # step inside bounds
    title="Approximation of parabola",
    labels=[
        "expert nn",
        "[32_16_8] nn",
        "f(x) = 3*x^2",
    ],  # labels for plots. Last label for true values
    true_data=(x_plot_data, y_plot_data),
)

str_sode = "y1*y2 y0(0)=0\n" + "-y0*y2 y1(0)=1\n" + "-0.5*y0*y1 y2(0)=1"
parsed = system_ode_from_string(str_sode)  # transform to list of strings
sode = SystemODE()
sode.prepare_equations(len(parsed), parsed)  # build functions for each equation
sode.solve((0, 3), 10)  # solve SODE on the interval
table = sode.build_table()
x_ode_data = table[:, :1]
y_ode_data = table[:, 1:]

for feature, value in zip(x_ode_data, y_ode_data):
    print(feature, value)

function = "3*x+2*y+4*z"
bounds = {
    "x": "0, 2, 1",  # x from 0 to 2 with step = 1
    "y": "0, 2, 2",  # y from 0 to 2 with step = 2
    "z": "0, 0, 1",  # z = 0
}  # 6 points

variables = str_eq_to_params(bounds)
eq_table = equation_solve(function, variables)

x_eq_data = eq_table[:, :-1]
y_eq_data = eq_table[:, -1:]

for feature, value in zip(x_eq_data, y_eq_data):
    print(feature, value)
