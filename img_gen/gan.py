import os
import time

from IPython import display
import matplotlib.pyplot as plt
import tensorflow as tf

from img_gen.models import (
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
        self.generator = img_generator(
            num_channels,
            width=width,
            height=height,
            conv_size=conv_size,
            norm_type=norm_type,
        )

        self.discriminator = discriminator(
            num_channels,
            width=width,
            height=height,
            conv_size=conv_size,
            norm_type=norm_type,
        )

        self.generator_optimizer = optimizer()
        self.discriminator_optimizer = optimizer()

        self.generator_losses = []
        self.discriminator_losses = []

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

    def train(
        self,
        dataset,
        epochs=60,
        noise_dim=100,
        num_examples_to_generate=16,
        batch_size=256,
    ):
        checkpoint_dir = "./training_checkpoints"
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator,
        )

        seed = tf.random.normal([num_examples_to_generate, noise_dim])

        # A. For each epoch, do the following:
        for epoch in range(epochs):
            start = time.time()
            # 1 - for each batch of the epoch,
            for image_batch in dataset:
                # 1.a -- run the custom train step function
                self.train_step(image_batch)

            # 2 - Produce images for the GIF as we go
            display.clear_output(wait=True)
            generate_and_save_images(self.generator, epoch + 1, seed)

            # 3 - Save the model every 5 epochs as a checkpoint,
            # which we will use later
            if (epoch + 1) % 5 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            # 4 - Print out the completed epoch no. and the time spent
            print(f"Time for epoch {epoch + 1} is {time.time() - start} sec")

        # B. Generate a final image after the training is completed
        display.clear_output(wait=True)
        generate_and_save_images(generator, epochs, seed)
