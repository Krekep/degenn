import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from degann.networks.topology.gan import GAN

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

gan = GAN(
    # Generator configuration (noise -> fake samples)
    generator_input_size=1,
    generator_block_size=[32, 32, 32],
    generator_output_size=1,
    generator_activation_func="leaky_relu",
    # Discriminator configuration (real/fake detection)
    discriminator_input_size=2,  # Input is [X, y] pairs
    discriminator_block_size=[32, 32, 32],
    discriminator_output_size=1,
    discriminator_activation_func="leaky_relu",
)

gan.custom_compile(
    generator_optimizer="Adam",
    generator_rate=0.0002,
    generator_loss_func="BinaryCrossentropy",
    discriminator_optimizer="Adam",
    discriminator_rate=0.0002,
    discriminator_loss_func="BinaryCrossentropy",
)

# Training loop with visualization
epochs = 1500
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")

    # Train on batches
    for batch in dataset:
        metrics = gan.train_step(batch)

    print(metrics)

    if (epoch + 1) % 250 == 0:
        # Generate comparison data
        n_vis = 500
        real_vis = (X_train[:n_vis], y_train[:n_vis])

        # Generate fake data with proper noise range
        vis_noise = tf.random.uniform((n_vis, 1), minval=0, maxval=1)
        fake_vis = (vis_noise.numpy(), gan(vis_noise).numpy())

        plot_comparison(real_vis, fake_vis, epoch + 1)

# Final comparison
final_noise = tf.random.uniform((1000, 1), minval=0, maxval=1)
final_fake = (final_noise.numpy(), gan(final_noise).numpy())
plot_comparison((X_train, y_train), final_fake, "Final")
