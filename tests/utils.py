import numpy as np
from tensorflow import keras


def array_compare(a, b, eps=-6):
    fl = True
    locality = 10**eps
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if isinstance(a[i], (list, np.ndarray)):
            if isinstance(b[i], (list, np.ndarray)):
                fl = fl & array_compare(a[i], b[i], eps=eps)
            else:
                fl = fl & False
        else:
            if isinstance(b[i], (list, np.ndarray)):
                fl = fl & False
            elif abs(a[i] - b[i]) > locality:
                fl = fl & False
    return fl


def init_params(weight_name: str = None, bias_name: str = None):
    return_param = []
    if weight_name is not None:
        weight_initializer = keras.initializers.get(weight_name)
        return_param.append(weight_initializer)
    if bias_name is not None:
        bias_initializer = keras.initializers.get(bias_name)
        return_param.append(bias_initializer)

    if len(return_param) == 0:
        return None
    if len(return_param) == 1:
        return return_param[0]
    return return_param


def file_compare(path1: str, path2: str) -> bool:
    f1 = open(path1)
    f2 = open(path2)
    f1_lines = f1.readlines()
    f2_lines = f2.readlines()

    if len(f1_lines) != len(f2_lines):
        return False
    for i in range(len(f1_lines)):
        if not f1_lines[i].__eq__(f2_lines[i]):
            return False
    return True


def create_main_func(
    type: str = "none", path: str = "./", import_file: str = "funcs", **kwargs
) -> str:
    """
    This function generates and returns the main function for C++ code
    need for tests

    Parametrs
    -------
    type: str
        type of main funtion
    path: str
        path to save result.txt
    kwargs

    Returns
    -------
    res: str
        main function
    """
    res = ""
    if type == "val_test":
        start_value = 0.003
        if "start_value" in kwargs:
            start_value = kwargs["start_value"]
        res = (
            f"""#include <fstream>
#include "{path + import_file}.cpp"\n\n"""
            + """int main() {\n\tfloat a[1] = { """
            + f"{start_value}"
            + """ };
"""
            + f'    std::ofstream result("{path}result.txt");'
            + """
    float* ans = feedforward(a);
    result << ans[0];
    result.close();
    return 0;
}
"""
        )
    elif type == "time_test":
        time_test_size = 2500
        if "time_test_size" in kwargs:
            time_test_size = kwargs["time_test_size"]
        res = (
            f"""#include <fstream>\n#include <chrono>
#include "{path + import_file}.cpp"\n\n"""
            + """int main() {\n\tauto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < """
            + f"{time_test_size}"
            + """; ++i) {
        float a[1] = { (float)(rand() % 20 + 1.0f) / (rand() % 20 + 5.0f) };
        feedforward(a);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
"""
            + f'    std::ofstream result("{path}result.txt");'
            + """
    result << duration.count();
    result.close();
    return 0;
}
"""
        )
    return res
