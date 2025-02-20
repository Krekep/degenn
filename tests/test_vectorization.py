import os
import pytest
import platform

from degann.networks.imodel import IModel
from degann.networks.cpp_utils import get_available_vectorized_levels
from tests.utils import create_main_func


@pytest.mark.parametrize(
    "input_size, shapes, act_funcs, output_size, test",
    [
        (
            1,
            [50, 50, 50, 50, 50],
            ["linear", "softsign", "hard_sigmoid", "softsign", "relu"],
            1,
            1,
        ),
        (
            1,
            [100, 100, 100, 100, 100, 100],
            ["hard_sigmoid", "relu", "softsign", "linear", "softsign", "softsign"],
            1,
            2,
        ),
        (1, [200, 200], ["linear", "hard_sigmoid"], 1, 3),
    ],
)
def test_value(input_size, shapes, act_funcs, output_size, test):
    folder_path = "./"
    path_for_export = ""
    if platform.system() == "Windows":
        current_file = os.path.realpath(__file__)
        folder_path = os.path.dirname(current_file) + "\\"
        for x in folder_path:
            if x == "\\":
                path_for_export += "\\"
            path_for_export += x
    shapes_size, vec_levels_size = len(shapes), 0
    vectorized_levels = {0: "none"}
    available_vectorized_levels = get_available_vectorized_levels()
    res = []

    if "sse" in available_vectorized_levels:
        vec_levels_size += 1
        vectorized_levels[vec_levels_size] = "sse"
    if "avx" in available_vectorized_levels:
        vec_levels_size += 1
        vectorized_levels[vec_levels_size] = "avx"
    if "avx512f" in available_vectorized_levels:
        vec_levels_size += 1
        vectorized_levels[vec_levels_size] = "avx512f"

    nn = IModel(input_size, shapes, output_size, act_funcs)
    vec_levels_size += 1
    for i in range(vec_levels_size):
        nn.export_to_cpp(
            f"{folder_path}funcs",
            vectorized_level=vectorized_levels[i],
        )
        main_code = open(f"{folder_path}main.cpp", "w")
        main_code.write(create_main_func(type="val_test", path=path_for_export))
        main_code.close()
        if i == 0:
            os.system(
                f"g++ -fno-tree-vectorize {folder_path}main.cpp -o {folder_path}main.exe"
            )
        else:
            os.system(
                f"g++ -m{vectorized_levels[i]} {folder_path}main.cpp -o {folder_path}main.exe"
            )

    for i in range(vec_levels_size):
        os.system(f"{folder_path}main.exe")
        result = open(f"{folder_path}result.txt", "r")
        res.append(result.read())
        result.close()

    os.remove(f"{folder_path}funcs.cpp")
    os.remove(f"{folder_path}funcs.hpp")
    os.remove(f"{folder_path}main.cpp")
    os.remove(f"{folder_path}main.exe")
    os.remove(f"{folder_path}result.txt")
    assert all(float(x) - float(res[0]) <= 0.001 for x in res)
