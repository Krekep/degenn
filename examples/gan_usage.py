import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from degann.networks import IModel
from degann.networks.topology.topology_parameters import (
    TensorflowDenseNetParams,
    GANTopologyParams,
)
from degann.networks.topology.compile_parameters import (
    SingleNetworkCompileParams,
    GANCompileParams,
)

# Fix random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


def generate_real_data(n_samples):
    X = np.random.uniform(0, 1, (n_samples, 1))
    y = np.sin(10 * X)
    return X.astype(np.float32), y.astype(np.float32)


def plot_comparison(real_data, fake_data, epoch=None):
    """Visualize real vs generated data"""
    plt.figure(figsize=(10, 6))

    # Plot real data
    plt.scatter(real_data[0], real_data[1], c="blue", label="Real Data", alpha=0.5)

    # Plot generated data
    plt.scatter(fake_data[0], fake_data[1], c="red", label="Generated Data", alpha=0.5)

    # Plot ideal relationship
    x = np.linspace(0, 1, 100)
    plt.plot(x, np.sin(10 * x), c="green", linestyle="--", label="Ideal: y = sin(10x)")

    plt.xlabel("X")
    plt.ylabel("y")
    plt.title(
        f'Data Distribution Comparison{"" if epoch is None else f" (Epoch {epoch})"}'
    )
    plt.legend()
    plt.grid(True)
    plt.show()


X_train, y_train = generate_real_data(2048)
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(64)

gen_config = TensorflowDenseNetParams(
    input_size=1, block_size=[32, 32, 32], output_size=1, activation_func="leaky_relu"
)
disc_config = TensorflowDenseNetParams(
    input_size=2, block_size=[32, 32, 32], output_size=1, activation_func="leaky_relu"
)
gan_params = GANTopologyParams(
    generator_params=gen_config, discriminator_params=disc_config
)
gan = IModel(gan_params)

gen_compile_config = SingleNetworkCompileParams(
    rate=0.0002,
    optimizer="Adam",
    loss_func="BinaryCrossentropy",
    metric_funcs=["MeanAbsoluteError"],
)
disc_compile_config = SingleNetworkCompileParams(
    rate=0.0002,
    optimizer="Adam",
    loss_func="BinaryCrossentropy",
    metric_funcs=["BinaryAccuracy"],
)
gan_compile_config = GANCompileParams(
    generator_params=gen_compile_config, discriminator_params=disc_compile_config
)
gan.compile(gan_compile_config)

log_dir = "examples/gan_usage_log"
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=0, embeddings_freq=0, update_freq="epoch"
)
gan.train(
    X_train, y_train, epochs=1500, mini_batch_size=64, callbacks=[tensorboard_callback]
)

final_noise = tf.random.uniform((1000, 1), minval=0, maxval=1)
final_fake = (final_noise.numpy(), gan.predict(final_noise))
plot_comparison((X_train, y_train), final_fake, "Final")
