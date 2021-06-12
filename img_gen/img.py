"""
This file defines utility functions for working with images.
"""

import tensorflow as tf

def random_crop(image, width=256, height=256):
    """
    Fetches a random width x height crop of the input image.
    """
    return tf.image.random_crop(image, size=[width, height, 3])


def nomralize(image):
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
        [width * scale, height * scale],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )

    # randomly crop back down to width x height
    image = random_crop(image, width, height)

    # random apply mirroring
    image = tf.image.random_flip_left_right(image)

    return image


def preprocess_training_image(image, width=256, height=256, jitter=True):
    """
    Preprocesses a training image by optionally applying a random jitter 
    and normalizing the image data.
    """
    if jitter:
        image = random_jitter(image, width, height)
    image = nomralize(image)
    return image


def image_similarity(image1, image2):
    """
    Measures the similarity of two images.
    """
    return tf.reduce_mean(tf.abs(image1 - image2))
