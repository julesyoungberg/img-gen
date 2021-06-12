import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
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

loss = BinaryCrossentropy(from_logits=True)


def downsample(filters, size, norm_type=None):
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
            strides=2,
            padding="same",
            kernel_initializer=initializer,
        )
    )

    if norm_type:
        if norm_type.lower() == "batchnorm":
            model.add(BatchNormalization())
        elif norm_type.lower() == "instancenorm":
            model.add(InstanceNormalization(axis=-1))

    model.add(LeakyReLU(0.2))

    return model


def upsample(filters, size, norm_type=None, apply_dropout=False):
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
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if norm_type:
        if norm_type.lower() == "batchnorm":
            model.add(BatchNormalization())
        elif norm_type.lower() == "instancenorm":
            model.add(InstanceNormalization())

    if apply_dropout:
        model.add(Dropout(0.5))

    model.add(ReLU())

    return model


def img_generator(output_channels, conv_size=4, width=256, height=256, norm_type=None):
    """
    Basic image generator network.
    """
    model = Sequential()
    model.add(Dense(2 * 2 * 1024, use_bias=False, input_shape=(100,)))

    if norm_type:
        if norm_type.lower() == "batchnorm":
            model.add(BatchNormalization())
        elif norm_type.lower() == "instancenorm":
            model.add(InstanceNormalization())

    model.add(LeakyReLU())
    model.add(Reshape((2, 2, 1024)))

    decoder_layers = [
        upsample(512, conv_size, norm_type, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, conv_size, norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, conv_size, norm_type),  # (bs, 16, 16, 1024)
        upsample(256, conv_size, norm_type),  # (bs, 32, 32, 512)
        upsample(128, conv_size, norm_type),  # (bs, 64, 64, 256)
        upsample(64, conv_size, norm_type),  # (bs, 128, 128, 128)
    ]

    for layer in decoder_layers:
        model = layer(model)

    initializer = tf.random_normal_initializer(0.0, 0.02)
    model = Conv2DTranspose(
        output_channels,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
    )(
        model
    )  # (bs, 256, 256, 3)

    return model


def unet_generator(
    output_channels, conv_size=4, width=256, height=256, norm_type="batchnorm"
):
    """
    Modified u-net generator model (https://arxiv.org/abs/1611.07004).
    """
    encoder_layers = [
        downsample(64, conv_size),  # (bs, 128, 128, 64)
        downsample(128, conv_size, norm_type),  # (bs, 64, 64, 128)
        downsample(256, conv_size, norm_type),  # (bs, 32, 32, 256)
        downsample(512, conv_size, norm_type),  # (bs, 16, 16, 512)
        downsample(512, conv_size, norm_type),  # (bs, 8, 8, 512)
        downsample(512, conv_size, norm_type),  # (bs, 4, 4, 512)
        downsample(512, conv_size, norm_type),  # (bs, 2, 2, 512)
        downsample(512, conv_size, norm_type),  # (bs, 1, 1, 512)
    ]

    decoder_layers = [
        upsample(512, conv_size, norm_type, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, conv_size, norm_type, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, conv_size, norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, conv_size, norm_type),  # (bs, 16, 16, 1024)
        upsample(256, conv_size, norm_type),  # (bs, 32, 32, 512)
        upsample(128, conv_size, norm_type),  # (bs, 64, 64, 256)
        upsample(64, conv_size, norm_type),  # (bs, 128, 128, 128)
    ]

    gen_input = Input(shape=(height, width, output_channels))
    gen = gen_input

    # downsampling through the model
    skips = []
    for down in encoder_layers:
        gen = down(gen)
        skips.append(gen)

    # reverse and remove first element
    skips = skips[::-1][1:]

    # upsampling and establishing the skip connections
    concat = Concatenate()
    for skip_layer, layer in zip(skips, decoder_layers):
        gen = layer(gen)
        gen = concat([gen, skip_layer])

    initializer = tf.random_normal_initializer(0.0, 0.02)
    gen = Conv2DTranspose(
        output_channels,
        conv_size,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
    )(
        gen
    )  # (bs, 256, 256, 3)

    return Model(inputs=gen_input, outputs=gen)


def discriminator(num_channels, conv_size=4, width=256, height=256, norm_type="batchnorm"):
    """
    PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
    """
    inp = Input(shape=(height, width, num_channels))
    d = inp

    d = downsample(64, conv_size)(d)  # (bs, 128, 128, 64)
    d = downsample(128, conv_size, norm_type)(d)  # (bs, 64, 64, 128)
    d = downsample(256, conv_size, norm_type)(d)  # (bs, 32, 32, 256)
    d = downsample(512, conv_size, norm_type)(d)  # (bs, 16, 16, 512)

    initializer = tf.random_normal_initializer(0.0, 0.02)
    d = Conv2D(1, conv_size, strides=1, kernel_initializer=initializer,)(
        d
    )  # (bs, 30, 30, 1)

    return Model(inputs=inp, outputs=d)


def discriminator_loss(real, generated):
    """
    Quantifies how well the disciminator is able to distinguish real
    images from fakes. It compares the disriminator's predictions on
    real images to an array of 1s, and the predictions on generated
    images to an array of 0s.
    """
    real_loss = loss(tf.ones_like(real), real)
    gen_loss = loss(tf.zeros_like(generated), generated)
    return real_loss * gen_loss * 0.5


def generator_loss(validity):
    """
    Quantifies how well the the generator was able to trick the
    discriminator. This can be measured by comparing the
    discriminator's predictions on generated images to
    and array of 1s.
    """
    return loss(tf.ones_like(validity), validity)


def optimizer():
    """
    Creates an optimizer.
    """
    return Adam(2e-4, beta_1=0.5)
