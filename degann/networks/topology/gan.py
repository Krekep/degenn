from typing import Optional

import tensorflow as tf

from degann.networks.topology.tf_densenet import TensorflowDenseNet
from degann.networks import metrics, optimizers

class GAN(tf.keras.Model):
    def __init__(
        self,
        generator_input_size: int = 1,
        generator_block_size: Optional[list] = None,
        generator_output_size: int = 1,
        generator_activation_func: str | list[str] = "leaky_relu",
        generator_weight_init=tf.random_uniform_initializer(minval=-1, maxval=1),
        generator_bias_init=tf.random_uniform_initializer(minval=-1, maxval=1),
        generator_is_debug: bool = False,
        discriminator_input_size: int = 1,
        discriminator_block_size: Optional[list] = None,
        discriminator_output_size: int = 1,
        discriminator_activation_func: str | list[str] = "leaky_relu",
        discriminator_weight_init=tf.random_uniform_initializer(minval=-1, maxval=1),
        discriminator_bias_init=tf.random_uniform_initializer(minval=-1, maxval=1),
        discriminator_is_debug: bool = False,
        **kwargs,
    ):
        self.noise_size = generator_input_size

        self.generator = TensorflowDenseNet(
            input_size=generator_input_size,
            block_size=generator_block_size,
            output_size=generator_output_size,
            activation_func=generator_activation_func,
            weight=generator_weight_init,
            biases=generator_bias_init,
            is_debug=generator_is_debug,
            **kwargs.get("generator", dict())
        )
        kwargs.pop("generator", None)

        self.discriminator = TensorflowDenseNet(
            input_size=discriminator_input_size,
            block_size=discriminator_block_size,
            output_size=discriminator_output_size,
            activation_func=discriminator_activation_func,
            weight=discriminator_weight_init,
            biases=discriminator_bias_init,
            is_debug=discriminator_is_debug,
            **kwargs.get("discriminator", dict())
        )
        kwargs.pop("discriminator", None)

        self.gan = None

        super(GAN, self).__init__(**kwargs)

    def custom_compile(
        self,
        generator_rate=1e-3,
        generator_optimizer="Adam",
        generator_loss_func="BinaryCrossentropy",
        generator_metric_funcs=[],
        generator_run_eagerly=False,
        discriminator_rate=1e-3,
        discriminator_optimizer="Adam",
        discriminator_loss_func="BinaryCrossentropy",
        discriminator_metric_funcs=[],
        discriminator_run_eagerly=False,
    ):
        # TODO: come up with a way to store and handle generator metrics

        #self.discriminator.trainable = False

        gan_input = tf.keras.Input(shape=(self.noise_size,))
        concat_layer = tf.keras.layers.Concatenate(axis=1)
        gan_output = self.discriminator(concat_layer(
            [gan_input, self.generator(gan_input)]
        ), training=False)
        self.gan = tf.keras.Model(gan_input, gan_output)

        opt = optimizers.get_optimizer(generator_optimizer)(
            learning_rate=generator_rate
        )
        self.gan.compile(
            loss=generator_loss_func,
            optimizer=opt,
            run_eagerly=generator_run_eagerly
        )

        #self.discriminator.trainable = True
        self.discriminator.custom_compile(
            discriminator_rate,
            discriminator_optimizer,
            discriminator_loss_func,
            discriminator_metric_funcs,
            discriminator_run_eagerly
        )

    def call(self, inputs, **kwargs):
        """
        Obtaining a generator response on the input data vector
        Parameters
        ----------
        inputs
        kwargs

        Returns
        -------

        """
        return self.generator(inputs, **kwargs)

    @tf.function
    def train_step(self, data):
        """
        Custom train step for GAN framework

        Parameters
        ----------
        data: tuple
            Pair of x and y (or dataset)
        Returns
        -------

        """
        # Unpack the data
        X, y = data
        batch_size = tf.shape(X)[0]

        # # Generate fake data
        # noise = tf.random.uniform(shape=(batch_size, self.noise_size))
        # generated_y = self.generator.call(noise, training=False)

        # # Prepare data for the discriminator
        # real_data = tf.concat([X, y], axis=1)
        # fake_data = tf.concat([noise, generated_y], axis=1)
        # combined_data = tf.concat([real_data, fake_data], axis=0)
        # combined_labels = tf.concat(
        #     [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        # )

        # # Train the discriminator
        # disc_metrics = self.discriminator.train_step(
        #     (combined_data, combined_labels)
        # )

        # # Train the generator
        # gen_loss = self.gan.train_step(
        #     (noise, tf.ones((batch_size, 1)))
        # )

        # disc_metrics = {f"disc_{k}": v for k, v in disc_metrics.items()}
        # gen_loss = {f"gen_{k}": v for k, v in gen_loss.items()}

        # return {**disc_metrics, **gen_loss}

        generated_x = tf.random.uniform(shape=(batch_size, self.noise_size))
        generated_y = self.generator(generated_x, training=False)

        real_data = tf.concat([X, y], axis=1)
        fake_data = tf.concat([generated_x, generated_y], axis=1)

        combined_data = tf.concat([real_data, fake_data], axis=0)
        combined_labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        with tf.GradientTape() as disc_tape:
            predictions = self.discriminator(combined_data, training=True)
            disc_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(combined_labels, predictions)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator.optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        with tf.GradientTape() as gen_tape:
            generated_y = self.generator(generated_x, training=True)
            fake_output = self.discriminator(tf.concat([generated_x, generated_y], axis=1), training=False)

            gen_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones((batch_size, 1)), fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gan.optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return gen_loss, disc_loss

    def set_name(self, new_name):
        self._name = new_name

    def __str__(self):
        return str(self.generator) + "\n\n" + str(self.discriminator)

    def to_dict(self, **kwargs):
        """
        Export neural network to dictionary

        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        res = {
            "generator": self.generator.to_dict(**kwargs.get("generator", dict())),
            "discriminator": self.discriminator.to_dict(**kwargs.get("discriminator", dict()))
        }

        return res

    def from_dict(self, config: dict):
        """
        Restore neural network from dictionary of params
        Parameters
        ----------
        config
        kwargs

        Returns
        -------

        """

        self.generator = TensorflowDenseNet()
        self.generator.from_dict(
            config["generator"]
        )

        self.discriminator = TensorflowDenseNet()
        self.discriminator.from_dict(
            config["discriminator"]
        )

    def export_to_cpp(
        self,
        path: str,
        array_type: str = "[]",
        path_to_compiler: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Export neural network as feedforward function on c++

        Parameters
        ----------
        path: str
            path to file with name, without extension
        array_type: str
            c-style or cpp-style ("[]" or "vector")
        path_to_compiler: str
            path to c/c++ compiler
        vectorized_level: str
            this is the vectorized level of C++ code
            if value is none, there is will standart code
            if value is auto, program will choose better availabale vectorization level
            and will use it
            if value is one of available vectorization levels (sse, avx, avx512f)
            then it level will be used in C++ code
        kwargs

        Returns
        -------

        """
        self.generator.export_to_cpp(
            path,
            array_type,
            path_to_compiler,
            **kwargs
        )
