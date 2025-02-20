from numbers import Number
from typing import Union, Tuple, List, Callable


def array1d_creator(
    elem_type: str,
) -> Callable[[str, int, Union[None, list[int | float], int | float]], str]:
    """
    Return function for creating c one-dimensional arrays (e.g. float[]) with specified type

    Parameters
    ----------
    elem_type:
        type of arrays

    Returns
    -------
    creator: Callable
        function for create arrays
    """

    def array1d_spec_type_creator(
        name: str,
        size: int,
        initial_value: Union[None, list[int | float], int | float] = None,
    ) -> str:
        """
        Create string representation of c one-dimensional array

        Parameters
        ----------
        name: str
            array name
        size: int
            array size
        initial_value: Union[None, list[int | float], int | float]
            initial value for array. Can be None, list of values (len should be equal to size) or single value

        Returns
        -------
        array: str
        """
        res = f"{elem_type} {name}[{size}]"
        if initial_value is not None:
            if isinstance(initial_value, list):
                if len(initial_value) == size:
                    res += " = {" + ", ".join(map(str, initial_value)) + "}"
                else:
                    raise ValueError("Incompatibility size of arrays")
            else:
                res += " = {" + ", ".join([str(initial_value)] * size) + "}"
        res += ";\n"
        return res

    return array1d_spec_type_creator


def array1d_heap_creator(elem_type: str) -> Callable[[str, int], str]:
    """
    Return function for creating c one-dimensional arrays on heap with specified type.
    E.g. (float*)malloc(size * sizeof(float))

    Parameters
    ----------
    elem_type:
        type of arrays

    Returns
    -------
    creator: Callable[[str, int], str]
        function for create arrays
    """

    def array1d_heap_spec_type_creator(name: str, size: int) -> str:
        """
        Create string representation of c one-dimensional array on heap

        Parameters
        ----------
        name: str
            array name
        size: int
            array size

        Returns
        -------
        array: str
        """
        res = (
            f"{elem_type}* {name} = ({elem_type}*)malloc({size} * sizeof({elem_type}))"
        )
        res += ";\n"
        return res

    return array1d_heap_spec_type_creator


def array2d_creator(
    elem_type: str,
) -> Callable[
    [str, int, int, Union[None, list[list[int | float]], int | float], bool], str
]:
    """
    Return function for creating c two-dimensional arrays (e.g. float[][]) with specified type

    Parameters
    ----------
    elem_type:
        type of arrays

    Returns
    -------
    creator: Callable
        function for create arrays
    """

    def array2d_spec_type_creator(
        name: str, size_x: int, size_y: int, initial_value=None, reverse=False
    ) -> str:
        """
        Create string representation of c two-dimensional array

        Parameters
        ----------
        name: str
            array name
        size_x: int
            first dimension size
        size_y: int
            second dimension  size
        initial_value: list[list]
            initial value for array. Can be None, list[list] of values (len should be equal to size) or single value
        reverse: bool
            rotate the matrix at a right angle counterclockwise

        Returns
        -------
        array: str
        """
        res = f"{elem_type} {name}[{size_x}][{size_y}]"
        if initial_value is not None:
            if isinstance(initial_value, list):
                if len(initial_value) == size_x and len(initial_value[0]) == size_y:
                    if reverse:
                        initial_value = [
                            [initial_value[i][j] for i in range(size_x)]
                            for j in range(size_y - 1, -1, -1)
                        ]
                    temp = "{ "
                    for i in range(size_x - 1):
                        temp += "{" + ", ".join(map(str, initial_value[i])) + "}, \n"
                    temp += "{" + ", ".join(map(str, initial_value[-1])) + "} }"
                    res += " = " + temp
                else:
                    raise ValueError("Incompatibility size of arrays")
            else:
                temp = "{ "
                for i in range(size_x - 1):
                    temp += "{" + ", ".join([str(initial_value)] * size_y) + "}, \n"
                temp += "{" + ", ".join([str(initial_value)] * size_y) + "} }"
                res += " = " + temp
        res += ";\n"
        return res

    return array2d_spec_type_creator


def vector1d_creator(elem_type: str):
    """
    Return function for creating cpp one-dimensional vectors (e.g. vector<float>) with specified type

    Parameters
    ----------
    elem_type:
        type of arrays

    Returns
    -------
    creator: Callable
        function for create arrays
    """

    def vector1d_spec_type_creator(
        name: str, size: int, initial_value: float = 0
    ) -> str:
        """
        Create string representation of cpp one dimensional vector

        Parameters
        ----------
        name: str
            vector name
        size: int
            vector size
        initial_value: float
            initial value for vector

        Returns
        -------
        vector: str
        """
        return f"std::vector<{elem_type}> {name}({size}, {initial_value});\n"

    return vector1d_spec_type_creator


def vector2d_creator(elem_type: str):
    """
    Return function for creating c two-dimensional vector (e.g. vector<vector<float>>) with specified type

    Parameters
    ----------
    elem_type:
        type of arrays

    Returns
    -------
    creator: Callable
        function for create arrays
    """

    def vector2d_spec_type_creator(
        name: str, size_x: int, size_y: int, initial_value=0
    ) -> str:
        """
        Create string representation of c two-dimensional vector

        Parameters
        ----------
        name: str
            vector name
        size_x: int
            first dimension size
        size_y: int
            second dimension  size
        initial_value: list
            initial value for vector

        Returns
        -------
        vector: str
        """
        return f"std::vector<std::vector<{elem_type}>> {name}({size_x}, std::vector<{type}>({size_y}, {initial_value}));\n"

    return vector2d_spec_type_creator


def transform_1dvector_to_array(
    elem_type: str, size: Union[int, str], vec_name: str, arr_name: str
) -> str:
    """
    Converts the one-dimensional vector given by name to the created array

    Parameters
    ----------
    elem_type: str
        Values type
    size: int | str
        Size of array and vector
    vec_name: str
        Vector name
    arr_name: str
        Array name

    Returns
    -------
    code: str
        Code to convert vector to array
    """
    res = f"""
    {elem_type} {arr_name}[{size}];

    for (int i = 0; i < {size}; i++)
    {{
        {arr_name}[i] = {vec_name}[i];
    }}\n
    """
    return res


def transform_1darray_to_vector(
    elem_type: str, size: Union[int, str], vec_name: str, arr_name: str
) -> str:
    """
    Converts the one-dimensional array given by name to the created vector

    Parameters
    ----------
    elem_type: str
        Values type
    size: int | str
        Size of array and vector
    vec_name: str
        Vector name
    arr_name: str
        Array name

    Returns
    -------
    code: str
        Code to convert array to vector
    """
    res = f"""
    std::vector<{elem_type}> {vec_name}({size});

    for (int i = 0; i < {size}; i++)
    {{
        {vec_name}[i] = {arr_name}[i];
    }}\n
    """
    return res


def copy_1darray_to_array(size: Union[int, str], in_name: str, out_name: str) -> str:
    """
    Copy one-dimensional array to other array

    Parameters
    ----------
    size: int | str
        Size of arrays
    in_name: str
        Name of the array from which values are copied
    out_name
        Name of the array into which the values are copied
    Returns
    -------
    code: str
        Code to copy array to array
    """
    res = f"""
    for (int i = 0; i < {size}; i++)
    {{
        {out_name}[i] = {in_name}[i];
    }}\n
    """
    return res


def fill_1d_array_by_list_short(
    elem_type: str, size: Union[int, str], arr_name: str, inter_name: str, source: list
) -> str:
    """
    Fills a one-dimensional array according to the passed list of values

    Parameters
    ----------
    elem_type: str
        Values type
    size: int | str
        Size of data
    arr_name: str
        Name of the array to fill
    inter_name: str
        Name of an array created for copying only
    source: list
        Values
    Returns
    -------
    code: str
        Code to fill array from list of values
    """
    res = f"""{elem_type} {inter_name}[{size}] = """
    temp = "{" + ", ".join(map(str, source)) + "};\n"
    res += temp
    res += f"""
    for (int i = 0; i < {size}; i++)
    {{
        {arr_name}[i] = {inter_name}[i];
    }}\n
    """

    return res


def fill_1d_array_by_list(arr_name: str, source: list) -> str:
    """
    Fills a one-dimensional array according to the passed list of values

    Parameters
    ----------
    arr_name: str
        Name of the array to fill
    source: list
        Values
    Returns
    -------
    code: str
        Code to fill array from list of values
    """
    res = ""
    for i, elem in enumerate(source):
        res += f"{arr_name}[{i}] = {elem};\n"

    return res


def fill_2d_array_by_list(arr_name: str, source: list) -> str:
    """
    Fills a two-dimensional array according to the passed list of values

    Parameters
    ----------
    arr_name: str
        Name of the array to fill
    source: list
        Values
    Returns
    -------
    code: str
        Code to fill array from list of values
    """
    res = ""
    for i, sub_arr in enumerate(source):
        for j, elem in enumerate(sub_arr):
            res += f"{arr_name}[{i}][{j}] = {elem};\n"

    return res


def feed_forward_step(
    left_name: str,
    left_size: int,
    right_name: str,
    right_size: int,
    weight_name: str,
    bias_name: str,
    activation_func: str,
) -> str:
    """
    This function builds the code for one step of the feed_forward (predict) method.
    In particular, it multiplies an array simulating a layer of neurons with a matrix of weights,
    adds an array of offsets and applies an activation function to the result

    Parameters
    ----------
    left_name: str
        Name of left layer
    left_size: str
        Size of left layer
    right_name: str
        Name of right layer
    right_size: str
        Size of right layer
    weight_name: str
        Name of weights matrix between left and right layers
    bias_name: str
        Name of biases array for right layer
    activation_func: str
        Name of activation for right layer
    Returns
    -------
    code: str
        Code to feed forward step
    """
    res = f"""
    for (int i = 0; i < {right_size}; i++)
    {{
        for (int j = 0; j < {left_size}; j++)
        {{
            {right_name}[i] += {weight_name}[j][i] * {left_name}[j];
        }}
        {right_name}[i] += {bias_name}[i];
        {activation_to_cpp_template(right_name + "[i]", activation_func)}
    }}
    """

    return res


def activation_to_cpp_template(name: str, activation_name: str) -> str:
    """
    Build code for activation function by function name and variable name

    Parameters
    ----------
    name: str
        Name of variable
    activation_name: str
        Name of activation func

    Returns
    -------
    c_activation: str
        Translated activation
    """
    d = {
        "linear": lambda x: f"{x} = {x};\n",
        "elu": lambda x: f"if ({x} >= 0) {x} = {x}; else {x} = 1.0 * (exp({x}) - 1);\n",
        "gelu": lambda x: f"{x} = 0.5 * {x} * (1 + tanh(sqrt(2 / 3.14159265) * ({x} + 0.044715 * {x} * {x} * {x})))",
        "relu": lambda x: f"{x} = max({x}, 0.0f);\n",
        "selu": lambda x: f"if ({x} >= 0) {x} = 1.05070098 * {x}; else {x} = 1.05070098 * 1.67326324 * (exp({x}) - 1);\n",
        "exponential": lambda x: f"{x} = exp({x});\n",
        "hard_sigmoid": lambda x: f"if ({x} < -2.5) {x} = 0; else if ({x} > 2.5) {x} = 1; else {x} = 0.2 * {x} + 0.5;\n",
        "sigmoid": lambda x: f"{x} = 1 / (1 + exp(-{x}));\n",
        "softplus": lambda x: f"{x} = log(exp({x}) + 1);\n",
        "softsign": lambda x: f"{x} = {x} / (abs({x}) + 1.0);\n",
        "swish": lambda x: f"{x} = {x} / (1 + exp(-{x}));\n",
        "tanh": lambda x: f"{x} = ((exp({x}) - exp(-{x}))/(exp({x}) + exp(-{x})));\n",
        "parabolic": lambda x: f"if ({x} >= 0) {x} = 0 + sqrt(2 * 1/5.0 * {x}); else {x} = 0 - sqrt(-2 * 1/5.0 * {x});\n",
    }

    return d[activation_name](name)
