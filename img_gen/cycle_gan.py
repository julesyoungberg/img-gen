"""
Code for building and optimizing CycleGANs.
"""

import os
import time

import keras_tuner as kt
from keras_tuner.engine.base_tuner import BaseTuner
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from img_gen.img import image_diff
from img_gen.models import (
    aggregate_losses,
    discriminator,
    discriminator_loss,
    generator_loss,
    resnet_generator,
    unet_generator,
    optimizer,
)
from img_gen.storage import save_figure


class CycleGAN:
    """
    The Cycle GAN architecture for image-to-image translation.
    """

    def __init__(
        self,
        num_channels=3,
        width=256,
        height=256,
        norm_type="instancenorm",  # batchnorm | instancenorm
        learning_rate=2e-4,
        loss_type="cross_entropy",  # cross_entropy | least_squares
        gen_type="unet",  # unet | resnet
        use_identity=True,
        gen_dropout=0.0,
        gen_conv_size=(3, 3),
        dis_loss_weight=0.5,
        dis_alpha=0.2,
        lmbd=10,
        use_cloud=False,
        cloud_bucket="img-gen-training",
        name="",
        show_images=False,
        save_images=False,
        save_models=False,
    ):
        self.num_channels = num_channels
        self.width = width
        self.height = height
        self.norm_type = norm_type
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        self.gen_type = gen_type
        self.use_identity = use_identity
        self.gen_dropout = gen_dropout
        self.gen_conv_size = gen_conv_size
        self.dis_loss_weight = dis_loss_weight
        self.dis_alpha = dis_alpha
        self.lmbd = lmbd
        self.use_cloud = use_cloud
        self.cloud_bucket = cloud_bucket
        self.show_images = show_images
        self.save_images = save_images
        self.save_models = save_models
        self.setup()
        self.name = name

    def setup(self):
        self.name = (
            "gen_type=" + self.gen_type + "__learning_rate=" + str(self.learning_rate)
        )
        self.name += "__loss_type=" + self.loss_type + "__gen_type=" + self.gen_type
        self.name += (
            "__use_identity="
            + str(self.use_identity)
            + "__gen_dropout="
            + str(self.gen_dropout)
        )
        self.name += "__gen_dropout=" + str(self.gen_dropout)
        self.name += "__gen_conv_size=" + str(self.gen_conv_size)
        self.name += "__dis_loss_weight=" + str(self.dis_loss_weight)
        self.name += "__dis_alpha=" + str(self.dis_alpha) + "__lmbd=" + str(self.lmbd)

        image_shape = (self.height, self.width, self.num_channels)

        # optimizers
        self.generator_g_optimizer = optimizer(learning_rate=self.learning_rate)
        self.generator_f_optimizer = optimizer(learning_rate=self.learning_rate)
        self.discriminator_x_optimizer = optimizer(learning_rate=self.learning_rate)
        self.discriminator_y_optimizer = optimizer(learning_rate=self.learning_rate)

        # losses
        self.generator_g_losses = []
        self.generator_f_losses = []
        self.discriminator_y_losses = []
        self.discriminator_x_losses = []
        self.generator_g_epoch_losses = []
        self.generator_f_epoch_losses = []
        self.discriminator_y_epoch_losses = []
        self.discriminator_x_epoch_losses = []

        if self.gen_type == "resnet":
            # generator G maps from image set X to Y
            self.generator_g = resnet_generator(
                image_shape=image_shape,
                conv_size=self.gen_conv_size,
                norm_type=self.norm_type,
                dropout=self.gen_dropout,
            )
            # generator F maps from image set Y to X
            self.generator_f = resnet_generator(
                image_shape=image_shape,
                conv_size=self.gen_conv_size,
                norm_type=self.norm_type,
                dropout=self.gen_dropout,
            )
        elif self.gen_type == "unet":
            # generator G maps from image set X to Y
            self.generator_g = unet_generator(
                image_shape=image_shape,
                conv_size=self.gen_conv_size,
                norm_type=self.norm_type,
                dropout=self.gen_dropout,
            )
            # generator F maps from image set Y to X
            self.generator_f = unet_generator(
                image_shape=image_shape,
                conv_size=self.gen_conv_size,
                norm_type=self.norm_type,
                dropout=self.gen_dropout,
            )
        else:
            raise ValueError("invalid gen_type")

        # discriminator x determines whether an image belongs to set X
        self.discriminator_x = discriminator(
            self.discriminator_x_optimizer,
            image_shape=image_shape,
            norm_type=self.norm_type,
            loss_weight=self.dis_loss_weight,
            alpha=self.dis_alpha,
        )
        # discriminator y determines whether an image belongs to set Y
        self.discriminator_y = discriminator(
            self.discriminator_y_optimizer,
            image_shape=image_shape,
            norm_type=self.norm_type,
            loss_weight=self.dis_loss_weight,
            alpha=self.dis_alpha,
        )

    def initialize_checkpoint_manager(self):
        """
        Initialize checkpoints and restore if possible.
        """
        checkpoint_path = "./checkpoints/"

        if self.use_cloud:
            checkpoint_path = os.path.join(
                "gs://", self.cloud_bucket, self.name, "checkpoints"
            )

        ckpt = tf.train.Checkpoint(
            generator_g=self.generator_g,
            generator_f=self.generator_f,
            discriminator_x=self.discriminator_x,
            discriminator_y=self.discriminator_y,
            generator_g_optimizer=self.generator_g_optimizer,
            generator_f_optimizer=self.generator_f_optimizer,
            discriminator_x_optimizer=self.discriminator_x_optimizer,
            discriminator_y_optimizer=self.discriminator_y_optimizer,
        )

        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Latest checkpoint restored!")

        return ckpt_manager

    @tf.function
    def calculate_losses(self, real_x, real_y):
        # 1. get the predictions
        # generator G translates X -> Y
        # generator F translates Y -> X
        fake_y = self.generator_g(real_x, training=True)
        cycled_x = self.generator_f(fake_y, training=True)
        fake_x = self.generator_f(real_y, training=True)
        cycled_y = self.generator_g(fake_x, training=True)

        # same x and same y are used for identity loss.
        id_x = self.generator_f(real_x, training=True)
        id_y = self.generator_g(real_y, training=True)

        # discriminate the real and generated results
        real_x_val = self.discriminator_x(real_x, training=True)
        real_y_val = self.discriminator_y(real_y, training=True)
        fake_x_val = self.discriminator_x(fake_x, training=True)
        fake_y_val = self.discriminator_y(fake_y, training=True)

        # 2. Calculate loss
        gen_g_adv_loss = generator_loss(fake_y_val, loss_type=self.loss_type)
        gen_f_adv_loss = generator_loss(fake_x_val, loss_type=self.loss_type)

        x_cycle_loss = image_diff(real_x, cycled_x)
        y_cycle_loss = image_diff(real_y, cycled_y)
        total_cycle_loss = (x_cycle_loss + y_cycle_loss) * self.lmbd

        # generator losses
        gen_g_loss = gen_g_adv_loss + total_cycle_loss
        gen_f_loss = gen_f_adv_loss + total_cycle_loss

        # optionally add identity loss
        if self.use_identity:
            gen_g_loss += image_diff(real_x, id_x) * 0.5 * self.lmbd
            gen_f_loss += image_diff(real_y, id_y) * 0.5 * self.lmbd

        # discriminator losses
        dis_x_loss = discriminator_loss(
            real_x_val, fake_x_val, loss_type=self.loss_type
        )
        dis_y_loss = discriminator_loss(
            real_y_val, fake_y_val, loss_type=self.loss_type
        )

        return gen_g_loss, gen_f_loss, dis_x_loss, dis_y_loss

    @tf.function
    def train_step(self, real_x, real_y):
        """
        Executes a single training step.
        Generates images, computes losses, computes gradients, updates models.
        """
        with tf.GradientTape(persistent=True) as tape:
            gen_g_loss, gen_f_loss, dis_x_loss, dis_y_loss = self.calculate_losses(
                real_x, real_y
            )

        # 3. calculate gradients for generator and discriminator
        gen_g_gradient = tape.gradient(gen_g_loss, self.generator_g.trainable_variables)
        gen_f_gradient = tape.gradient(gen_f_loss, self.generator_f.trainable_variables)
        dis_x_gradient = tape.gradient(
            dis_x_loss, self.discriminator_x.trainable_variables
        )
        dis_y_gradient = tape.gradient(
            dis_y_loss, self.discriminator_y.trainable_variables
        )

        # 4. apply gradients to optimizer to update model
        self.generator_g_optimizer.apply_gradients(
            zip(gen_g_gradient, self.generator_g.trainable_variables)
        )
        self.generator_f_optimizer.apply_gradients(
            zip(gen_f_gradient, self.generator_f.trainable_variables)
        )
        self.discriminator_x_optimizer.apply_gradients(
            zip(dis_x_gradient, self.discriminator_x.trainable_variables)
        )
        self.discriminator_y_optimizer.apply_gradients(
            zip(dis_y_gradient, self.discriminator_y.trainable_variables)
        )

        # 5. save current losses
        return gen_g_loss, gen_f_loss, dis_x_loss, dis_y_loss

    def aggregate_losses(self):
        self.generator_g_losses.append(aggregate_losses(self.generator_g_epoch_losses))
        self.generator_f_losses.append(aggregate_losses(self.generator_f_epoch_losses))
        self.discriminator_x_losses.append(
            aggregate_losses(self.discriminator_x_epoch_losses)
        )
        self.discriminator_y_losses.append(
            aggregate_losses(self.discriminator_y_epoch_losses)
        )
        self.generator_g_epoch_losses = []
        self.generator_f_epoch_losses = []
        self.discriminator_x_epoch_losses = []
        self.discriminator_y_epoch_losses = []

    def print_losses(self):
        print("gen_f: ", self.generator_f_losses[-1], end=", ")
        print("gen_g: ", self.generator_g_losses[-1], end=", ")
        print(
            "dis_x: ",
            self.discriminator_x_losses[-1],
            end=", ",
        )
        print("dis_y: ", self.discriminator_y_losses[-1])

    def generate_images(self, test_x, test_y, epoch=None, typ="test"):
        """
        Generates image from the test input.
        """
        if not self.show_images and not self.save_images:
            return

        # sample images
        img_shape = (self.height, self.width, self.num_channels)
        x = next(iter(test_x.shuffle(1))).numpy().reshape(img_shape)
        y = next(iter(test_y.shuffle(1))).numpy().reshape(img_shape)

        # get predictions for those images
        shape = (1, self.height, self.width, self.num_channels)
        y_hat = self.generator_g.predict(x.reshape(shape)).reshape(img_shape)
        x_hat = self.generator_f.predict(y.reshape(shape)).reshape(img_shape)

        if not self.show_images:
            plt.ioff()

        # plot images
        fig = plt.figure(figsize=(12, 12))

        images = [x, y_hat, y, x_hat]

        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.imshow(images[i] * 0.5 + 0.5)
            plt.axis("off")

        plt.tight_layout()

        if self.show_images:
            plt.show()

        if self.save_images:
            filename = typ + "_generations"
            if epoch is not None:
                filename += "_" + str(epoch)
            filename += ".png"

            path = "./images/" + filename
            if self.use_cloud:
                path = f"{self.name}/images/{filename}"
                save_figure(fig, path)
            else:
                plt.savefig(path)

        if not self.show_images:
            plt.ion()

    def save_current_models(self, save_dir="models"):
        if self.use_cloud:
            os.path.join("gs://", self.cloud_bucket, self.name, save_dir)

        self.generator_g.save(os.path.join(save_dir, "generator_g"))
        self.generator_f.save(os.path.join(save_dir, "generator_f"))
        self.discriminator_x.save(os.path.join(save_dir, "discriminator_x"))
        self.discriminator_y.save(os.path.join(save_dir, "discriminator_y"))

    def load_models(self, save_dir="models"):
        if self.use_cloud:
            os.path.join("gs://", self.cloud_bucket, self.name, save_dir)

        self.generator_g.load_weights(os.path.join(save_dir, "generator_g"))
        self.generator_f.load_weights(os.path.join(save_dir, "generator_f"))
        self.discriminator_x.load_weights(os.path.join(save_dir, "discriminator_x"))
        self.discriminator_y.load_weights(os.path.join(save_dir, "discriminator_y"))

    def train(
        self,
        train_x,
        train_y,
        test_x=None,
        test_y=None,
        epochs=10,
        checkpoints=True,
        on_epoch_end=None,
    ):
        """
        Train the networks.
        """
        # tf.config.run_functions_eagerly(True)
        # tf.compat.v1.disable_eager_execution()
        print(f"x examples: {len(train_x)}, y examples: {len(train_y)}")

        if checkpoints:
            ckpt_manager = self.initialize_checkpoint_manager()

        if test_x is not None and test_y is not None:
            self.generate_images(test_x, test_y, epoch=-1)

        shape = (1, self.width, self.height, self.num_channels)

        for epoch in range(epochs):
            start = time.time()

            num_samples = len(train_x)
            percent_done = 0
            prev_done = 0

            data = enumerate(tf.data.Dataset.zip((train_x, train_y)))
            print("data len: ", len(data))

            print(f"epoch: {epoch} ", end="")

            # run the train_step algorithm for each image
            for k, (real_x, real_y) in data:
                gen_g_loss, gen_f_loss, dis_x_loss, dis_y_loss = self.train_step(
                    tf.reshape(real_x, shape), tf.reshape(real_y, shape)
                )

                self.generator_g_epoch_losses.append(gen_g_loss)
                self.generator_f_epoch_losses.append(gen_f_loss)
                self.discriminator_x_epoch_losses.append(dis_x_loss)
                self.discriminator_y_epoch_losses.append(dis_y_loss)

                # visual feedback
                percent_done = int(k / num_samples * 100)
                while prev_done < percent_done:
                    print(".", end="")
                    prev_done += 1

            print(f" time taken: {time.time() - start}s")

            self.aggregate_losses()
            self.print_losses()

            self.generate_images(train_x, train_y, typ="train", epoch=epoch)

            if test_x is not None and test_y is not None:
                self.generate_images(test_x, test_y, typ="test", epoch=epoch)

            # save checkpoint every 5 epochs
            if checkpoints and (epoch + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                print(f"saving checkpoint at {ckpt_save_path}")

            if on_epoch_end is not None:
                loss = np.array(
                    [
                        self.generator_g_losses[-1],
                        self.generator_f_losses[-1],
                        self.discriminator_x_losses[-1],
                        self.discriminator_y_losses[-1],
                    ]
                ).mean()
                on_epoch_end(epoch, loss)

        if self.save_models:
            self.save_current_models()

    def fit(self, train_x, train_y, epochs=2, checkpoints=True, on_epoch_end=None):
        self.train(
            train_x,
            train_y,
            epochs=epochs,
            checkpoints=checkpoints,
            on_epoch_end=on_epoch_end,
        )

    def plot_losses(self):
        fig = plt.figure(figsize=(12, 12))

        plt.plot(self.generator_g_losses, label="gen_g")
        plt.plot(self.generator_f_losses, label="gen_f")
        plt.plot(self.discriminator_x_losses, label="dis_x")
        plt.plot(self.discriminator_y_losses, label="dis_y")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("CycleGAN Losses")

        plt.legend()
        plt.show()

        if self.save_images:
            path = "./images/losses.png"

            if self.use_cloud:
                path = f"{self.cloud_bucket}/{self.name}/losses.png"
                save_figure(fig, path)
            else:
                plt.savefig(path)

    def scores(self, test_x, test_y):
        # tf.config.run_functions_eagerly(True)

        test_x = tf.data.Dataset(test_x)
        test_y = tf.data.Dataset(test_y)

        shape = (1, self.height, self.width, self.num_channels)

        gen_g_losses = np.array([])
        gen_f_losses = np.array([])
        dis_x_losses = np.array([])
        dis_y_losses = np.array([])

        # calculae losses for each image
        for (raw_x, raw_y) in tf.data.Dataset.zip((test_x, test_y)):
            real_x = tf.reshape(raw_x, shape)
            real_y = tf.reshape(raw_y, shape)

            gen_g_loss, gen_f_loss, dis_x_loss, dis_y_loss = self.calculate_losses(
                real_x, real_y
            )

            # save the losses
            gen_g_losses = np.append(gen_g_losses, gen_g_loss)
            gen_f_losses = np.append(gen_f_losses, gen_f_loss)
            dis_x_losses = np.append(dis_x_losses, dis_x_loss)
            dis_y_losses = np.append(dis_y_losses, dis_y_loss)

        return (
            gen_g_losses.mean(),
            gen_f_losses.mean(),
            dis_x_losses.mean(),
            dis_y_losses.mean(),
        )

    def score(self, test_x, test_y):
        (gen_g_loss, gen_f_loss, dis_x_loss, dis_y_loss) = self.scores(test_x, test_y)
        return np.array([gen_g_loss, gen_f_loss, dis_x_loss, dis_y_loss]).mean()


PARAMETERS = [
    "norm_type",
    "loss_type",
    "gen_type",
    "use_identity",
    "gen_dropout",
    "gen_conv_size",
    "dis_loss_weight",
    "lmbd",
    "dis_alpha",
]


def build_model(hp, **params):
    """Builds an optimizable cycle gan."""
    norm_type = hp.Choice("norm_type", ["batchnorm", "instancenorm"])
    loss_type = hp.Choice("loss_type", ["cross_entropy", "least_squares"])
    gen_type = hp.Choice("gen_type", ["unet", "resnet"])
    use_identity = hp.Choice("use_identity", [False, True])
    gen_dropout = hp.Float("gen_dropout", 0.0, 0.5, default=0.0)
    gen_conv_size = hp.Choice("gen_conv_size", [(3, 3), (4, 4)], default=(3, 3))
    dis_loss_weight = hp.Float("dis_loss_weight", 0.5, 1.0, default=1.0)
    lmbd = hp.Int("lmbd", 1, 15, default=10)
    learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3)
    dis_alpha = hp.Float("dis_alpha", 0.1, 0.7, default=0.2)

    cycle_gan = CycleGAN(
        norm_type=norm_type,
        loss_type=loss_type,
        gen_type=gen_type,
        use_identity=use_identity,
        gen_dropout=gen_dropout,
        gen_conv_size=gen_conv_size,
        dis_loss_weight=dis_loss_weight,
        lmbd=lmbd,
        learning_rate=learning_rate,
        dis_alpha=dis_alpha,
        **params,
    )

    return cycle_gan


class GANTuner(BaseTuner):
    def __init__(
        self,
        oracle,
        hypermodel,
        **kwargs,
    ):
        super().__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)

    # https://github.com/keras-team/keras-tuner/blob/b69f320c8cb4453d6f4d0eb00f3f71a78bda55c5/keras_tuner/engine/tuner.py#L247
    def on_epoch_end(self, trial, model, epoch, logs=None):
        """Called at the end of an epoch.
        Args:
            trial: A `Trial` instance.
            model: A Keras `Model`.
            epoch: The current epoch number.
            logs: Dict. Metrics for this epoch. This should include
              the value of the objective for this epoch.
        """
        self.save_model(trial.trial_id, model, step=epoch)
        # Report intermediate metrics to the `Oracle`.
        status = self.oracle.update_trial(trial.trial_id, metrics=logs, step=epoch)
        trial.status = status
        if trial.status == "STOPPED":
            model.stop_training = True

    def run_trial(self, trial, X, y):
        hp = trial.hyperparameters

        model = self.hypermodel.build(trial.hyperparameters)
        # model = build_model(trial.hyperparameters)

        def on_epoch_end(epoch, loss):
            self.on_epoch_end(trial, model, epoch, logs={"loss": loss})

        model.fit(X, y, on_epoch_end=on_epoch_end)

    def save_model(self, trial_id, model, step=0):
        model.save_current_models(self.get_trial_dir(trial_id))

    def load_model(self, trial):
        model = self.hypermodelo.build(trial.hyperparameters)
        model.load_models(self.get_trial_dir(trial.trial_id))


def find_optimal_cycle_gan(
    train_x,
    train_y,
    test_x,
    test_y,
    epochs=40,
    checkpoints=True,
    directory="optimization_results",
    project_name="cycle_gan",
    **params,
):
    tuner = GANTuner(
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective("loss", "min"), max_trials=2
        ),
        hypermodel=build_model,
        directory=directory,
        project_name=project_name,
    )

    tuner.search(train_x, train_y)

    best_params = tuner.get_best_hyperparameters()[0]
    print("optimal hyper parameters:")
    for param in PARAMETERS:
        print(f"    - {param}: {best_params.get(param)}")

    # build and train optimal model
    cycle_gan = build_model(
        best_params,
        show_images=False,
        save_images=True,
        save_models=True,
        **params,
    )

    cycle_gan.train(
        train_x, train_y, test_x, test_y, epochs=epochs, checkpoints=checkpoints
    )

    return cycle_gan
