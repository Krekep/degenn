from dataclasses import dataclass, field
from typing import Union, Any, Optional
import tensorflow as tf

from degann.networks.topology.utils import TuningMetadata


@dataclass
class BaseTopologyParams:
    """
    Base class for common neural network topology parameters.

    This class holds the core parameters that define the structure of a neural network,
    such as input size, block (hidden layer) sizes, and output size.

    Attributes:
        input_size (int): Size of the input vector.
        block_size (List[int]): List of neuron counts for each hidden layer.
        output_size (int): Size of the output vector.
        name (str): Name identifier for the network.
        net_type (str): Type identifier for the network (e.g., "DenseNet").
        is_debug (bool): Flag to enable debugging mode.
    """

    tuning_metadata: dict[str, TuningMetadata] = field(default_factory=dict)

    input_size: int = 1
    block_size: list[int] = field(default_factory=list)
    output_size: int = 1
    name: str = "net"
    net_type: str = "DenseNet"
    is_debug: bool = False


@dataclass
class TensorflowDenseNetParams(BaseTopologyParams):
    """
    Parameters for a fully-connected (dense) neural network topology.

    Attributes:
        activation_func (Union[str, List[str]]): The activation function(s) to use in the network.
        weight (Any): Initializer for the network's weights. (Default: RandomUniform between -1 and 1)
        biases (Any): Initializer for the network's biases. (Default: RandomUniform between -1 and 1)
    """

    activation_func: Union[str, list[str]] = "sigmoid"
    weight: Any = field(
        default_factory=lambda: tf.random_uniform_initializer(minval=-1, maxval=1)
    )
    biases: Any = field(
        default_factory=lambda: tf.random_uniform_initializer(minval=-1, maxval=1)
    )

    def __post_init__(self):
        self.net_type = "DenseNet"


@dataclass(
    kw_only=True
)  # kw_only used to bypass the attribute organisation of @dataclass
class GANTopologyParams(BaseTopologyParams):
    """
    Parameters for a GAN (Generative Adversarial Network) topology.

    This topology consists of two neural networks:
      - A generator network
      - A discriminator network

    Attributes:
        generator_params (TensorflowDenseNetParams): Configuration parameters for the generator.
        discriminator_params (TensorflowDenseNetParams): Configuration parameters for the discriminator.
    """

    generator_params: TensorflowDenseNetParams
    discriminator_params: TensorflowDenseNetParams

    def __post_init__(self):
        # Set the overall GAN configuration based on the generator and discriminator settings.
        # The overall input size is taken from the generator.
        self.input_size = self.generator_params.input_size
        # The overall block_size is constructed by concatenating:
        #   1. The generator's block_size,
        #   2. A bridging layer equal to the discriminator's input_size,
        #   3. The discriminator's block_size.
        self.block_size = (
            self.generator_params.block_size
            + [self.discriminator_params.input_size]
            + self.discriminator_params.block_size
        )
        # The output size is defined by the discriminator's output.
        self.output_size = self.discriminator_params.output_size
        self.net_type = "GAN"
