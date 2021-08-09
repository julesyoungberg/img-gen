"""
Helper functions for creating and working with tensorflow models needed for constructing GANs.
This code is a large mix of pieces from various tutorials, that are all referenced, along with
the tutorial's results in ../notebooks/tutorials.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Add,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    Input,
    Layer,
    LeakyReLU,
    ReLU,
    Reshape,
    ZeroPadding2D,
)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization

loss = BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)


def downsample(
    filters,
    size=(3, 3),
    strides=(2, 2),
    norm_type=None,
    leaky=False,
    alpha=0.2,
    activation="relu",
):
    """
    Downsamples an input.
    Applies Convolution, normalization, and leaky relu activation.
    """
    initializer = tf.random_normal_initializer(0.0, 0.02)
    model = Sequential()
    model.add(
        Conv2D(
            filters,
            size,
            strides=strides,
            padding="same",
            kernel_initializer=initializer,
        )
    )

    if norm_type == "batchnorm":
        model.add(BatchNormalization())
    elif norm_type == "instancenorm":
        model.add(InstanceNormalization(axis=-1))

    if activation == "relu":
        if leaky:
            model.add(LeakyReLU(alpha))
        else:
            model.add(ReLU())
    elif activation == "tanh":
        model.add(Activation("tanh"))

    return model


def upsample(
    filters,
    size=(3, 3),
    strides=2,
    norm_type=None,
    dropout=0.0,
    activation="relu",
):
    """
    Upsamples an input.
    Applies inverse convolution, normalization, dropout, and relu normalization.
    """
    initializer = tf.random_normal_initializer(0.0, 0.02)

    model = Sequential()
    model.add(
        Conv2DTranspose(
            filters,
            size,
            strides=strides,
            padding="same",
            kernel_initializer=initializer,
            # use_bias=False,
        )
    )

    if norm_type == "batchnorm":
        model.add(BatchNormalization())
    elif norm_type == "instancenorm":
        model.add(InstanceNormalization(axis=-1))

    if dropout > 0:
        model.add(Dropout(dropout))

    if activation == "relu":
        model.add(ReLU())
    elif activation == "tanh":
        model.add(Activation("tanh"))

    return model


def residual_block(inpt, filters=256, conv_size=(3, 3), norm_type="instancenorm"):
    """
    Creates a residual block for downsampling an image.
    based on: https://machinelearningmastery.com/how-to-develop-cyclegan-models-from-scratch-with-keras/
    """
    initializer = tf.random_normal_initializer(0.0, 0.02)

    # layer 1
    r = Conv2D(filters, conv_size, padding="same", kernel_initializer=initializer)(inpt)
    if norm_type == "batchnorm":
        r = BatchNormalization()(r)
    elif norm_type == "instancenorm":
        r = InstanceNormalization(axis=-1)(r)

    # layer 2
    r = Conv2D(filters, conv_size, padding="same", kernel_initializer=initializer)(r)
    if norm_type == "batchnorm":
        r = BatchNormalization()(r)
    elif norm_type == "instancenorm":
        r = InstanceNormalization(axis=-1)(r)

    # add input to output to create residual block
    # g(x) = f(x) + x
    r = Add()([r, inpt])

    return r


def img_generator(num_channels=3, conv_size=(3, 3), norm_type=None, dropout=0.5):
    """
    Basic image generator network.
    """
    model = Sequential()
    model.add(Dense(2 * 2 * 1024, use_bias=False, input_shape=(100,)))

    if norm_type == "batchnorm":
        model.add(BatchNormalization())
    elif norm_type == "instancenorm":
        model.add(InstanceNormalization())

    model.add(LeakyReLU())
    model.add(Reshape((2, 2, 1024)))

    decoder_layers = [
        upsample(
            512,
            size=conv_size,
            norm_type=norm_type,
            dropout=dropout,
        ),  # (bs, 4, 4, 1024)
        upsample(
            512,
            size=conv_size,
            norm_type=norm_type,
            dropout=dropout,
        ),  # (bs, 8, 8, 1024)
        upsample(512, size=conv_size, norm_type=norm_type),  # (bs, 16, 16, 1024)
        upsample(256, size=conv_size, norm_type=norm_type),  # (bs, 32, 32, 512)
        upsample(128, size=conv_size, norm_type=norm_type),  # (bs, 64, 64, 256)
        upsample(64, size=conv_size, norm_type=norm_type),  # (bs, 128, 128, 128)
    ]

    for layer in decoder_layers:
        model = layer(model)

    # last layer
    gen = upsample(num_channels, size=conv_size, activation="tanh")(
        gen
    )  # (bs, 256, 256, 3)

    return model


def unet_generator(
    image_shape=(256, 256, 3),
    conv_size=(3, 3),
    norm_type="instancenorm",
    dropout=0.0,
):
    """
    U-net generator model.
    """
    gen_input = Input(shape=image_shape)

    encoder_layers = [
        downsample(64, size=conv_size),  # (bs, 128, 128, 64)
        downsample(128, size=conv_size, norm_type=norm_type),  # (bs, 64, 64, 128)
        downsample(256, size=conv_size, norm_type=norm_type),  # (bs, 32, 32, 256)
        downsample(512, size=conv_size, norm_type=norm_type),  # (bs, 16, 16, 512)
        downsample(512, size=conv_size, norm_type=norm_type),  # (bs, 8, 8, 512)
        downsample(512, size=conv_size, norm_type=norm_type),  # (bs, 4, 4, 512)
        downsample(512, size=conv_size, norm_type=norm_type),  # (bs, 2, 2, 512)
        downsample(512, size=conv_size, norm_type=norm_type),  # (bs, 1, 1, 512)
    ]

    decoder_layers = [
        upsample(
            512,
            size=conv_size,
            norm_type=norm_type,
            dropout=dropout,
        ),  # (bs, 2, 2, 1024)
        upsample(
            512,
            size=conv_size,
            norm_type=norm_type,
            dropout=dropout,
        ),  # (bs, 4, 4, 1024)
        upsample(
            512,
            size=conv_size,
            norm_type=norm_type,
            dropout=dropout,
        ),  # (bs, 8, 8, 1024)
        upsample(512, size=conv_size, norm_type=norm_type),  # (bs, 16, 16, 1024)
        upsample(256, size=conv_size, norm_type=norm_type),  # (bs, 32, 32, 512)
        upsample(128, size=conv_size, norm_type=norm_type),  # (bs, 64, 64, 256)
        upsample(64, size=conv_size, norm_type=norm_type),  # (bs, 128, 128, 128)
    ]

    gen = gen_input

    # downsampling through the model
    skips = []
    for layer in encoder_layers:
        gen = layer(gen)
        skips.append(gen)

    # reverse and remove first element
    skips = skips[::-1][1:]

    concat = Concatenate()

    # upsampling and establishing the skip connections
    for skip_layer, layer in zip(skips, decoder_layers):
        gen = concat([layer(gen), skip_layer])

    # last layer
    gen = upsample(image_shape[2], size=conv_size, activation="tanh")(
        gen
    )  # (bs, 256, 256, 3)

    return Model(gen_input, gen)


def resnet_generator(
    image_shape=(256, 256, 3),
    conv_size=(3, 3),
    num_res_blocks=9,
    norm_type="instancenorm",
    dropout=0.0,
):
    """
    Modified Res-Net (https://arxiv.org/abs/1611.07004).
    based on: https://machinelearningmastery.com/how-to-develop-cyclegan-models-from-scratch-with-keras/
    """
    inpt = Input(shape=image_shape)

    ## Encoder layers
    ##
    # c7s1-64
    g = downsample(
        64,
        (7, 7),
        strides=(1, 1),
        norm_type=norm_type,
    )(inpt)
    # d128
    g = downsample(128, size=conv_size, norm_type=norm_type)(g)
    # d256
    g = downsample(256, size=conv_size, norm_type=norm_type)(g)
    # R256
    for _ in range(num_res_blocks):
        g = residual_block(g, filters=256, conv_size=conv_size, norm_type=norm_type)

    ## Decoder layers
    ##
    # u128
    g = upsample(
        128,
        size=conv_size,
        norm_type=norm_type,
        dropout=dropout,
    )(g)
    # u64
    g = upsample(
        64,
        size=conv_size,
        norm_type=norm_type,
        dropout=dropout,
    )(g)
    # c7s1-3
    g = downsample(
        image_shape[2], (7, 7), strides=(1, 1), norm_type=norm_type, activation="tanh"
    )(g)

    return Model(inpt, g)


def discriminator(
    opt,
    image_shape=(256, 256, 3),
    conv_size=(4, 4),
    norm_type="instancenorm",
    alpha=0.2,
):
    """
    PatchGan discriminator model.
    """
    inpt = Input(shape=image_shape)

    d = downsample(64, size=conv_size, leaky=True, alpha=alpha)(
        inpt
    )  # (bs, 128, 128, 64)
    d = downsample(128, size=conv_size, norm_type=norm_type, leaky=True, alpha=alpha)(
        d
    )  # (bs, 64, 64, 128)
    d = downsample(256, size=conv_size, norm_type=norm_type, leaky=True, alpha=alpha)(
        d
    )  # (bs, 32, 32, 256)
    d = downsample(512, size=conv_size, norm_type=norm_type, leaky=True, alpha=alpha)(
        d
    )  # (bs, 16, 16, 512)

    d = downsample(1, size=conv_size, strides=(1, 1), activation=None)(
        d
    )  # (bs, 30, 30, 1)

    return Model(inpt, d)


def discriminator_loss_cross_entropy(
    real, generated, flip_labels=False, soft_labels=False
):
    if flip_labels:
        real_labels = tf.zeros_like(real)
        gen_labels = tf.ones_like(generated)

        if soft_labels:
            real_labels = real_labels + tf.random.uniform(shape=[real.shape[0]]) * 0.1
            gen_labels = gen_labels - tf.random.unifom(shape=[generated.shape[0]]) * 0.1
    else:
        real_labels = tf.ones_like(real)
        gen_labels = tf.zeros_like(generated)

        if soft_labels:
            real_labels = real_labels - tf.random.uniform(shape=[real.shape[0]]) * 0.1
            gen_labels = (
                gen_labels + tf.random.uniform(shape=[generated.shape[0]]) * 0.1
            )

    real_loss = loss(real_labels, real)
    gen_loss = loss(gen_labels, generated)

    return real_loss + gen_loss


def discriminator_loss_least_squares(
    real, generated, flip_labels=False, soft_labels=False
):
    if flip_labels:
        generated = generated - 1

        if soft_labels:
            real = real - tf.random.uniform(shape=[real.shape[0]]) * 0.1
            generated = generated + tf.random.uniform(shape=[generated.shape[0]]) * 0.1
    else:
        real = real - 1

        if soft_labels:
            real = real + tf.random.uniform(shape=[real.shape[0]]) * 0.1
            generated = generated - tf.random.uniform(shape=[generated.shape[0]]) * 0.1

    return tf.math.reduce_mean(tf.math.square(real)) + tf.math.reduce_mean(
        tf.math.square(generated)
    )


def discriminator_loss(
    real, generated, loss_type="least_squares", flip_labels=False, soft_labels=False
):
    """
    Quantifies how well the disciminator is able to distinguish real
    images from fakes. It compares the disriminator's predictions on
    real images to an array of 1s, and the predictions on generated
    images to an array of 0s.
    """
    if loss_type == "cross_entropy":
        return discriminator_loss_cross_entropy(
            real, generated, flip_labels=flip_labels, soft_labels=soft_labels
        )

    if loss_type == "least_squares":
        return discriminator_loss_least_squares(
            real, generated, flip_labels=flip_labels, soft_labels=soft_labels
        )

    return None


def generator_loss_cross_entropy(validity, flip_labels=False):
    labels = tf.zeros_like(validity) if flip_labels else tf.ones_like(validity)
    return loss(labels, validity)


def generator_loss_least_squares(validity, flip_labels=False):
    x = validity if flip_labels else validity - 1
    return tf.math.reduce_mean(tf.math.square(x))


def generator_loss(validity, loss_type="least_squares", flip_labels=False):
    """
    Quantifies how well the the generator was able to trick the
    discriminator. This can be measured by comparing the
    discriminator's predictions on generated images to
    and array of 1s.
    """
    if loss_type == "cross_entropy":
        return generator_loss_cross_entropy(validity, flip_labels=flip_lables)

    if loss_type == "least_squares":
        return generator_loss_least_squares(validity, flip_labels=flip_labels)

    return None


def optimizer(learning_rate=2e-4):
    """
    Creates an optimizer.
    """
    return Adam(learning_rate=learning_rate, beta_1=0.5)


def aggregate_losses(losses):
    return np.array(losses).mean()
