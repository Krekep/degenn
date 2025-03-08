from typing import Optional, List

import tensorflow as tf

from degann.networks.topology.tf_densenet import TensorflowDenseNet
from degann.networks.topology.topology_parameters import GANTopologyParams
from degann.networks.topology.compile_parameters import GANCompileParams
from degann.networks import metrics, optimizers, losses


class GAN(tf.keras.Model):
    def __init__(self, config: GANTopologyParams, **kwargs):
        self.input_size = config.input_size
        self.block_size = config.block_size
        self.output_size = config.output_size

        self.trained_time = {"train_time": 0.0, "epoch_time": [], "predict_time": 0}

        generator_kwargs = kwargs.pop("generator", dict())
        self.generator = TensorflowDenseNet(
            config.generator_params,
            **generator_kwargs,
        )

        discriminator_kwargs = kwargs.pop("discriminator", dict())
        self.discriminator = TensorflowDenseNet(
            config.discriminator_params,
            **discriminator_kwargs,
        )

        self.d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")

        self.gen_metrics: List[tf.keras.metrics.Metric] = []
        self.disc_metrics: List[tf.keras.metrics.Metric] = []

        super(GAN, self).__init__(**kwargs)

    @property
    def metrics(self):
        """
        Returns a list of all metrics to be reset between epochs.
        """
        base_metrics = [self.d_loss_tracker, self.g_loss_tracker]
        return base_metrics + self.disc_metrics + self.gen_metrics

    def custom_compile(self, config: GANCompileParams):
        """
        Configures the model for training

        Parameters
        ----------
        config: GANCompileParams
            parameters for compilation containing learning rate, optimizer,
            loss function and metrics for the generator and discriminator

        Returns
        -------

        """
        super(GAN, self).compile()

        gan_input = tf.keras.Input(shape=(self.input_size,))
        concat_layer = tf.keras.layers.Concatenate(axis=1)
        gan_output = self.discriminator(
            concat_layer([gan_input, self.generator(gan_input, training=True)]),
            training=False,
        )
        self.gan: tf.keras.Model = tf.keras.Model(gan_input, gan_output)

        opt = optimizers.get_optimizer(config.generator_params.optimizer)(
            learning_rate=config.generator_params.rate
        )
        loss = losses.get_loss(config.generator_params.loss_func)

        self.gan.compile(
            optimizer=opt,
            loss=loss,
            run_eagerly=config.generator_params.run_eagerly,
        )
        self.gen_metrics = [
            metrics.get_metric(metric)
            for metric in config.generator_params.metric_funcs
        ]

        self.discriminator.custom_compile(config.discriminator_params)
        self.disc_metrics = [
            metrics.get_metric(metric)
            for metric in config.discriminator_params.metric_funcs
        ]

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

        generated_x = tf.random.uniform(shape=(batch_size, self.input_size))
        generated_y = self.generator(generated_x, training=False)

        real_data = tf.concat([X, y], axis=1)
        fake_data = tf.concat([generated_x, generated_y], axis=1)

        combined_data = tf.concat([real_data, fake_data], axis=0)
        combined_labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        with tf.GradientTape() as disc_tape:
            predictions = self.discriminator(combined_data, training=True)
            disc_loss = self.discriminator.compute_loss(
                y=combined_labels, y_pred=predictions
            )

        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )
        self.discriminator.optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )

        with tf.GradientTape() as gen_tape:
            fake_output = self.gan(generated_x, training=True)
            gen_loss = self.gan.compute_loss(
                y=tf.ones((batch_size, 1)), y_pred=fake_output
            )

        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        self.gan.optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables)
        )

        # Update metrics
        self.d_loss_tracker.update_state(disc_loss)
        self.g_loss_tracker.update_state(gen_loss)

        for metric in self.disc_metrics:
            metric.update_state(combined_labels, predictions)

        y_fake = self.generator(X)
        for metric in self.gen_metrics:
            metric.update_state(y, y_fake)

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        """
        Custom test (evaluation) step for GAN.
        """
        X, y = data
        batch_size = tf.shape(X)[0]

        generated_x = tf.random.uniform(shape=(batch_size, self.input_size))
        generated_y = self.generator(generated_x, training=False)

        real_data = tf.concat([X, y], axis=1)
        fake_data = tf.concat([generated_x, generated_y], axis=1)

        g_loss = self.gan.compute_loss(
            y=tf.ones((batch_size, 1)), y_pred=self.discriminator(fake_data)
        )

        combined_data = tf.concat([real_data, fake_data], axis=0)
        combined_labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        predictions = self.discriminator(combined_data, training=False)
        d_loss = self.discriminator.compute_loss(y=combined_labels, y_pred=predictions)

        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)

        for metric in self.disc_metrics:
            metric.update_state(combined_labels, predictions)

        y_fake = self.generator(X, training=False)
        for metric in self.gen_metrics:
            metric.update_state(y, y_fake)

        return {m.name: m.result() for m in self.metrics}

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
            "discriminator": self.discriminator.to_dict(
                **kwargs.get("discriminator", dict())
            ),
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

        self.generator.from_dict(config["generator"])
        self.discriminator.from_dict(config["discriminator"])

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
        self.generator.export_to_cpp(path, array_type, path_to_compiler, **kwargs)

    @property
    def get_activations(self) -> List:
        """
        Get list of activations functions for each layer

        Returns
        -------
        activation: list
        """
        return (
            self.generator.get_activations + ["|"] + self.discriminator.get_activations
        )

    def get_loss_names(self):
        return "g_loss", "d_loss"
