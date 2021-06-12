"""
This file defines utility functions for working with images.
"""

import tensorflow as tf


def random_crop(image, width=256, height=256):
    """
    Fetches a random width x height crop of the input image.
    """
    return tf.image.random_crop(image, size=[width, height, 3])


def normalize(image):
    """
    Normalizes image data from [0, 255] to [-1, 1].
    """
    return tf.cast(image, tf.float32) / 127.5 - 1


@tf.function
def random_jitter(image, width=256, height=256):
    """
    Applies random jittering to the input image.
    """
    scale = 1.12

    # resize 1.12
    image = tf.image.resize(
        image,
        [int(width * scale), int(height * scale)],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )

    # randomly crop back down to width x height
    image = random_crop(image, width, height)

    # random apply mirroring
    image = tf.image.random_flip_left_right(image)

    return image


def preprocess_image(image, channels=3, width=256, height=256, jitter=False):
    """
    Preprocesses a training image by optionally applying a random jitter
    and normalizing the image data.
    """
    if jitter:
        image = random_jitter(image, width, height)
    else:
        image = tf.image.resize(image, (height, width))
    image = normalize(image)
    return tf.reshape(image, (1, height, width, channels))


def preprocess_images(
    images, width=256, height=256, jitter=False, buffer_size=1000, batch_size=1
):
    """
    Preprocesses, shuffles, and batches a set of images.
    """

    def f(image, _=None):
        return preprocess_image(image, width=width, height=height, jitter=jitter)

    return images.map(f)  # .cache().shuffle(buffer_size).batch(batch_size)


def image_similarity(image1, image2):
    """
    Measures the similarity of two images.
    """
    return tf.reduce_mean(tf.abs(image1 - image2))
