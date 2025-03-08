from dataclasses import dataclass, field
from typing import Optional

from degann.networks.topology.utils import TuningMetadata


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

    tuning_metadata: dict[str, TuningMetadata] = field(default_factory=dict)


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
    metric_funcs: list[str] = field(
        default_factory=lambda: [
            "root_mean_squared_error",
        ]
    )
    run_eagerly: bool = False

    def get_losses(self):
        return [self.loss_func]

    def get_optimizers(self):
        return [self.optimizer]

    def add_eval_metric(self, metric: str):
        self.metric_funcs.append(metric)


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

    def get_losses(self):
        return (
            self.generator_params.get_losses() + self.discriminator_params.get_losses()
        )

    def get_optimizers(self):
        return (
            self.generator_params.get_optimizers()
            + self.discriminator_params.get_optimizers()
        )

    def add_eval_metric(self, metric: str):
        self.generator_params.add_eval_metric(metric)
