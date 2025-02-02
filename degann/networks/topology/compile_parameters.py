from dataclasses import dataclass, field
from typing import List

@dataclass
class BaseCompileParams:
    """
    Base class for compilation parameters applicable to neural network topologies.

    This class is intended to serve as a foundation for more specialized compile parameter
    configurations (e.g., for single-network or GAN topologies).

    Attributes:
        (This base class does not define any fields by itself but acts as a base
        for inheritance.)
    """
    pass

@dataclass
class SingleNetworkCompileParams(BaseCompileParams):
    """
    Compilation parameters for a single-network topology.

    Attributes:
        rate (float): Learning rate for the optimizer.
        optimizer (str): Name of the optimizer.
        loss_func (str): Loss function to use.
        metric_funcs (List[str]): List of metric function names.
        run_eagerly (bool): Whether to run eagerly.
    """
    rate: float = 1e-2
    optimizer: str = "SGD"
    loss_func: str = "MeanSquaredError"
    metric_funcs: List[str] = field(default_factory=lambda: [
        "MeanSquaredError", "MeanAbsoluteError", "MeanSquaredLogarithmicError"
    ])
    run_eagerly: bool = False

@dataclass(kw_only=True)
class GANCompileParams(BaseCompileParams):
    """
    Compilation parameters for a GAN (Generative Adversarial Network) topology.

    This configuration includes separate compile settings for the generator and discriminator.

    Attributes:
        generator_params (SingleNetworkCompileParams): Compile parameters for the generator.
        discriminator_params (SingleNetworkCompileParams): Compile parameters for the discriminator.
    """
    generator_params: SingleNetworkCompileParams
    discriminator_params: SingleNetworkCompileParams
