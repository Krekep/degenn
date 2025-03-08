from itertools import product
from typing import Optional, Tuple, List

from degann.networks import imodel

act_to_hex = {
    "elu": "0",
    "relu": "1",
    "gelu": "2",
    "selu": "3",
    "exponential": "4",
    "linear": "5",
    "sigmoid": "6",
    "hard_sigmoid": "7",
    "swish": "8",
    "tanh": "9",
    "softplus": "a",
    "softsign": "b",
    "parabolic": "c",
    "leaky_relu": "d",
}
hex_to_act = {v: k for k, v in act_to_hex.items()}

alph_n_full = "0123456789abcdef"
alph_n_div3 = "0369cf"
alph_n_div2 = "02468ace"
alph_n_div4 = "048c"

alphabet_activations_cut = "0689"
alphabet_activations = "0123456789abcd"
default_alphabet: list[str] = [
    "".join(elem) for elem in product(alph_n_full, alphabet_activations)
]


def encode(nn: imodel.IModel, offset: Optional[int] = None) -> str:
    """
    Encode neural network to string

    Parameters
    ----------
    nn: IModel
        Neural network for coding
    offset: int
        Minimum possible layer size
    Returns
    -------
    code: str
        Encoded neural network
    """
    blocks = nn.get_shape
    activations = nn.get_activations
    res = ""

    offset = min(blocks) if offset is None else offset
    for layer, act in zip(blocks, activations):
        if act == "|":
            res += "|" + hex(layer - offset)[2:] + "|"
        else:
            res += hex(layer - offset)[2:] + act_to_hex[act]

    return res


def decode(s: str, block_size: int = 1, offset: int = 0) -> Tuple[List[int], List[str]]:
    """
    Decode neural network from string as a pair of shape and activations

    Parameters
    ----------
    s: str
        Neural network as code
    block_size: int
        Number of letters allocated to encode the size of one layer
    offset: int
        Minimum possible layer size
    Returns
    -------
    network: tuple[list[int], list[str]]
        Pair of shape and activations for neural network
    """
    blocks = []
    activations = []

    i = 0
    while i < len(s):
        block_end = i + block_size

        if s[i] == "|":
            i += 1
            blocks.append(int(s[i:block_end], 16) + offset)
            activations.append("|")
        else:
            blocks.append(int(s[i:block_end], 16) + offset)
            activations.append(hex_to_act[s[block_end]])

        # Skip past the next separator "|" or move to the next block
        i = block_end + 1

    return blocks, activations
