import os
import time

from IPython import display
import matplotlib.pyplot as plt
import tensorflow as tf

from img_gen.models import (
    aggregate_losses,
    discriminator,
    discriminator_loss,
    generator_loss,
    img_generator,
    optimizer,
)


def generate_and_save_images(model, epoch, test_input):
    # Notice 'training' is set to False
    # This is so all layers run in inference mode (batchnorm).
    # 1 - Generate images
    predictions = model(test_input, training=False)
    # 2 - Plot the generated images
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 128.5, cmap="gray")
        plt.axis("off")
    # 3 - Save the generated images
    plt.savefig("image_at_epoch_{:04d}.png".format(epoch))
    plt.show()


class GAN:
    """
    A basic image generator gan.
    """

    def __init__(
        self,
        num_channels=3,
        width=256,
        height=256,
        conv_size=4,
        norm_type="instancenorm",
    ):
        self.generator_optimizer = optimizer()
        self.discriminator_optimizer = optimizer()

        self.generator_losses = []
        self.discriminator_losses = []

        self.generator = img_generator(
            num_channels=num_channels,
            conv_size=conv_size,
            norm_type=norm_type,
        )

        self.discriminator = discriminator(
            self.discriminator_optimizer,
            image_shape=(height, width, num_channels),
            conv_size=conv_size,
            norm_type=norm_type,
        )

    def initialize_checkpoint_manager(self):
        checkpoint_dir = "./training_checkpoints"
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint_manager = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator,
        )
        return checkpoint_manager, checkpoint_prefix

    @tf.function
    def train_step(self, images, noise_dim=100, batch_size=256):
        """
        Executes a single training step.
        Generates images, computes losses, computes gradients, updates models.
        """
        # 1 - Create random noise to feed it into the model
        # for image generation
        noise = tf.random.normal([batch_size, noise_dim])

        # 2 - Generate images and calculate loss values
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            dis_loss = discriminator_loss(real_output, fake_output)

        # 3 - Calculate gradients using loss values and model variables.
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        dis_gradients = dis_tape.gradient(
            dis_loss, self.discriminator.trainable_variables
        )

        # 4 - Process gradients and run the optimizer
        self.generator_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(dis_gradients, self.discriminator.trainable_variables)
        )

        # 5. Save the current losses
        self.generator_losses.append(gen_loss)
        self.discriminator_losses.append(dis_loss)

    def aggregate_losses(self, n):
        self.generator_losses = aggregate_losses(self.generator_losses, n)
        self.discriminator_losses = aggregate_losses(self.discriminator_losses, n)

    def print_losses(self):
        print("gen: ", self.generator_losses[-1].numpy(), end=", ")
        print("dis: ", self.discriminator_losses[-1].numpy())

    def train(
        self,
        dataset,
        epochs=60,
        noise_dim=100,
        num_examples_to_generate=16,
        checkpoints=True,
    ):
        tf.config.run_functions_eagerly(True)

        if checkpoints:
            ckpt_manager, ckpt_prefix = self.initialize_checkpoint_manager()

        seed = tf.random.normal([num_examples_to_generate, noise_dim])

        # generate_and_save_images(self.generator, 0, seed)

        # A. For each epoch, do the following:
        for epoch in range(epochs):
            print(f"epoch: {epoch} ", end="")
            start = time.time()

            num_samples = len(dataset)
            percent_done = 0
            prev_done = 0

            # 1 - for each batch of the epoch,
            for k, image_batch in enumerate(dataset):
                # 1.a -- run the custom train step function
                self.train_step(image_batch)

                percent_done = int(k / num_samples * 100)
                while prev_done < percent_done:
                    print(".", end="")
                    prev_done += 1

            print(f" time taken: {time.time() - start}s")

            self.aggregate_losses()
            self.print_losses()

            # 2 - Produce images for the GIF as we go
            display.clear_output(wait=True)
            generate_and_save_images(self.generator, epoch + 1, seed)

            # 3 - Save the model every 5 epochs as a checkpoint,
            # which we will use later
            if checkpoints and (epoch + 1) % 5 == 0:
                ckpt_manager.save(file_prefix=ckpt_prefix)

        # B. Generate a final image after the training is completed
        display.clear_output(wait=True)
        generate_and_save_images(self.generator, epochs, seed)

    def plot_losses(self):
        plt.plot(self.generator_losses, label="generator")
        plt.plot(self.discriminator_losses, label="discriminator")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("GAN Losses")

        plt.legend()
        plt.show()
