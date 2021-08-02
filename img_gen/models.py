import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Activation,
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
    apply_dropout=False,
    dropout=0.5,
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

    if apply_dropout:
        model.add(Dropout(dropout))

    if activation == "relu":
        model.add(ReLU())
    elif activation == "tanh":
        model.add(Activation("tanh"))

    return model


def residual_block(inpt, filters=256, size=(3, 3), norm_type="instancenorm"):
    """
    Creates a residual block for downsampling an image.
    """
    initializer = tf.random_normal_initializer(0.0, 0.02)

    # layer 1
    r = Conv2D(filters, size, padding="same", kernel_initializer=initializer)(inpt)
    if norm_type == "batchnorm":
        r = BatchNormalization()(r)
    elif norm_type == "instancenorm":
        r = InstanceNormalization(axis=-1)(r)

    # layer 2
    r = Conv2D(filters, size, padding="same", kernel_initializer=initializer)(inpt)
    if norm_type == "batchnorm":
        r = BatchNormalization()(r)
    elif norm_type == "instancenorm":
        r = InstanceNormalization(axis=-1)(r)

    # add input to output to create residual block
    # g(x) = f(x) + x
    r = Concatenate()([r, inpt])

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
            apply_dropout=True,
            dropout=dropout,
        ),  # (bs, 4, 4, 1024)
        upsample(
            512,
            size=conv_size,
            norm_type=norm_type,
            apply_dropout=True,
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
    apply_dropout=True,
    dropout=0.5,
):
    """
    Modified u-net generator model (https://arxiv.org/abs/1611.07004).
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
            apply_dropout=apply_dropout,
            dropout=dropout,
        ),  # (bs, 2, 2, 1024)
        upsample(
            512,
            size=conv_size,
            norm_type=norm_type,
            apply_dropout=apply_dropout,
            dropout=dropout,
        ),  # (bs, 4, 4, 1024)
        upsample(
            512,
            size=conv_size,
            norm_type=norm_type,
            apply_dropout=apply_dropout,
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
    image_shape=(256, 256, 3), num_res_blocks=9, norm_type="instancenorm"
):
    """
    Modifier Res-Net (https://arxiv.org/abs/1611.07004).
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
    g = downsample(128, (3, 3), norm_type=norm_type)(g)
    # d256
    g = downsample(256, (3, 3), norm_type=norm_type)(g)
    # R256
    for _ in range(num_res_blocks):
        g = residual_block(g, filters=256, norm_type=norm_type)

    ## Decoder layers
    ##
    # u128
    g = upsample(128, (3, 3), norm_type=norm_type)(g)
    # u64
    g = upsample(64, (3, 3), norm_type=norm_type)(g)
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
    loss_weight=0.5,
    alpha=0.2,
):
    """
    Modified PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
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

    d = downsample(1, conv_size, strides=(1, 1), activation=None)(d)  # (bs, 30, 30, 1)

    model = Model(inpt, d)
    model.compile(loss="mse", optimizer=opt, loss_weights=[loss_weight])
    return model


def discriminator_loss_cross_entropy(real, generated):
    real_loss = loss(tf.ones_like(real), real)
    gen_loss = loss(tf.zeros_like(generated), generated)
    return (real_loss + gen_loss) * 0.5


def discriminator_loss_least_squares(real, generated):
    return tf.math.reduce_mean(tf.math.square(real - 1) + tf.math.square(generated))


def discriminator_loss(real, generated, loss_type="least_squares"):
    """
    Quantifies how well the disciminator is able to distinguish real
    images from fakes. It compares the disriminator's predictions on
    real images to an array of 1s, and the predictions on generated
    images to an array of 0s.
    """
    if loss_type == "cross_entropy":
        return discriminator_loss_cross_entropy(real, generated)

    if loss_type == "least_squares":
        return discriminator_loss_least_squares(real, generated)

    return None


def generator_loss_cross_entropy(validity):
    return loss(tf.ones_like(validity), validity)


def generator_loss_least_squares(validity):
    return tf.math.reduce_mean(tf.math.square(validity - 1))


def generator_loss(validity, loss_type="least_squares"):
    """
    Quantifies how well the the generator was able to trick the
    discriminator. This can be measured by comparing the
    discriminator's predictions on generated images to
    and array of 1s.
    """
    if loss_type == "cross_entropy":
        return generator_loss_cross_entropy(validity)

    if loss_type == "least_squares":
        return generator_loss_least_squares(validity)

    return None


def optimizer(learning_rate=2e-4):
    """
    Creates an optimizer.
    """
    return Adam(learning_rate=learning_rate, beta_1=0.5)


def aggregate_losses(losses, n):
    """
    Aggregate the last n losses by averaging.
    """
    aggregated = losses[: len(losses) - n]
    print("aggregated:", aggregated)
    new_losses = np.array(losses[len(losses) - n :])
    print("new losses:", new_losses)
    mean = new_losses.mean()
    print("mean: ", mean)
    aggregated.append(mean)
    print("new aggregated:", aggregated)
    return aggregated
