import time

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
        gen_type="unet",
        use_identity=True,
        gen_dropout=0.5,
        gen_apply_dropout=False,
        dis_loss_weight=0.5,
        dis_alpha=0.2,
    ):
        self.loss_type = loss_type
        self.num_channels = num_channels
        self.width = width
        self.height = height
        self.use_identity = use_identity

        image_shape = (height, width, num_channels)

        # optimizers
        self.generator_g_optimizer = optimizer(learning_rate=learning_rate)
        self.generator_f_optimizer = optimizer(learning_rate=learning_rate)
        self.discriminator_x_optimizer = optimizer(learning_rate=learning_rate)
        self.discriminator_y_optimizer = optimizer(learning_rate=learning_rate)

        # losses
        self.generator_g_losses = []
        self.generator_f_losses = []
        self.discriminator_y_losses = []
        self.discriminator_x_losses = []

        if gen_type == "resnet":
            # generator G maps from image set X to Y
            self.generator_g = resnet_generator(
                image_shape=image_shape,
                norm_type=norm_type,
            )
            # generator F maps from image set Y to X
            self.generator_f = resnet_generator(
                image_shape=image_shape,
                norm_type=norm_type,
            )
        elif gen_type == "unet":
            # generator G maps from image set X to Y
            self.generator_g = unet_generator(
                image_shape=image_shape,
                norm_type=norm_type,
                apply_dropout=gen_apply_dropout,
                dropout=gen_dropout,
            )
            # generator F maps from image set Y to X
            self.generator_f = unet_generator(
                image_shape=image_shape,
                norm_type=norm_type,
                apply_dropout=gen_apply_dropout,
                dropout=gen_dropout,
            )
        else:
            raise ValueError("invalid gen_type")

        # discriminator x determines whether an image belongs to set X
        self.discriminator_x = discriminator(
            self.discriminator_x_optimizer,
            image_shape=image_shape,
            norm_type=norm_type,
            loss_weight=dis_loss_weight,
            alpha=dis_alpha,
        )
        # discriminator y determines whether an image belongs to set Y
        self.discriminator_y = discriminator(
            self.discriminator_y_optimizer,
            image_shape=image_shape,
            norm_type=norm_type,
            loss_weight=dis_loss_weight,
            alpha=dis_alpha,
        )

    def initialize_checkpoint_manager(self):
        """
        Initialize checkpoints and restore if possible.
        """
        checkpoint_path = "./checkpoints/train"

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
    def calculate_losses(self, real_x, real_y, lmbd=10):
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
        total_cycle_loss = (x_cycle_loss + y_cycle_loss) * lmbd

        # generator losses
        gen_g_loss = gen_g_adv_loss + total_cycle_loss
        gen_f_loss = gen_f_adv_loss + total_cycle_loss

        # optionally add identity loss
        if self.use_identity:
            gen_g_loss += image_diff(real_x, id_x) * 0.5 * lmbd
            gen_f_loss += image_diff(real_y, id_y) * 0.5 * lmbd

        # discriminator losses
        dis_x_loss = discriminator_loss(
            real_x_val, fake_x_val, loss_type=self.loss_type
        )
        dis_y_loss = discriminator_loss(
            real_y_val, fake_y_val, loss_type=self.loss_type
        )

        return gen_g_loss, gen_f_loss, dis_x_loss, dis_y_loss

    @tf.function
    def train_step(self, real_x, real_y, lmbd=10):
        """
        Executes a single training step.
        Generates images, computes losses, computes gradients, updates models.
        """
        with tf.GradientTape(persistent=True) as tape:
            gen_g_loss, gen_f_loss, dis_x_loss, dis_y_loss = self.calculate_losses(
                real_x, real_y, lmbd=lmbd
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
        self.generator_g_losses.append(gen_g_loss)
        self.generator_f_losses.append(gen_f_loss)
        self.discriminator_x_losses.append(dis_x_loss)
        self.discriminator_y_losses.append(dis_y_loss)

    def aggregate_losses(self, n):
        self.generator_g_losses = aggregate_losses(self.generator_g_losses, n)
        self.generator_f_losses = aggregate_losses(self.generator_f_losses, n)
        self.discriminator_x_losses = aggregate_losses(self.discriminator_x_losses, n)
        self.discriminator_y_losses = aggregate_losses(self.discriminator_y_losses, n)

    def print_losses(self):
        print("gen_g: ", self.generator_g_losses[-1].numpy(), end=", ")
        print("gen_f: ", self.generator_f_losses[-1].numpy(), end=", ")
        print("dis_x: ", self.discriminator_x_losses[-1].numpy(), end=", ")
        print("dis_y: ", self.discriminator_y_losses[-1].numpy())

    def generate_images(self, test_X, test_y):
        """
        Generates image from the test input.
        """
        # sample images
        x = next(iter(test_X.shuffle(1000))).numpy()
        y = next(iter(test_y.shuffle(1000))).numpy()

        # get predictions for those images
        shape = (1, self.height, self.width, self.num_channels)
        y_hat = self.generator_g.predict(x.reshape(shape))
        x_hat = self.generator_f.predict(y.reshape(shape))

        # plot images
        plt.figure(figsize=(12, 12))

        images = [x[0], y_hat[0], y[0], x_hat[0]]

        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.imshow(images[i] * 0.5 + 0.5)
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    def train(self, train_X, train_y, test_X, test_y, epochs=40, checkpoints=True):
        """
        Train the networks.
        """
        tf.config.run_functions_eagerly(True)

        if checkpoints:
            ckpt_manager = self.initialize_checkpoint_manager()

        shape = (1, self.height, self.width, self.num_channels)

        # self.generate_images(test_x, test_y)

        for epoch in range(epochs):
            print(f"epoch: {epoch} ", end="")
            start = time.time()

            num_samples = len(train_X)
            percent_done = 0
            prev_done = 0

            data = enumerate(tf.data.Dataset.zip((train_X, train_y)))
            for k, (real_x, real_y) in data:
                self.train_step(tf.reshape(real_x, shape), tf.reshape(real_y, shape))

                percent_done = int(k / num_samples * 100)
                while prev_done < percent_done:
                    print(".", end="")
                    prev_done += 1

            print(f" time taken: {time.time() - start}s")

            self.aggregate_losses(num_samples)
            self.print_losses()
            self.generate_images(test_X, test_y)

            if checkpoints and (epoch + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                print(f"saving checkpoint at {ckpt_save_path}")

    def plot_losses(self):
        plt.plot(self.generator_g_losses, label="gen_g")
        plt.plot(self.generator_f_losses, label="gen_f")
        plt.plot(self.discriminator_x_losses, label="dis_x")
        plt.plot(self.discriminator_y_losses, label="dis_y")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("CycleGAN Losses")

        plt.legend()
        plt.show()

    def score(self, test_x, test_y, lmbd=10):
        tf.config.run_functions_eagerly(True)

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
                real_x, real_y, lmbd=lmbd
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
