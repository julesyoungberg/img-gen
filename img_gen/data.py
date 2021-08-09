import os, sys, tarfile
import urllib.request

from openimages.download import download_images
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow_datasets as tfds

from img_gen.img import preprocess_images


def load_openimages(input_labels=[], output_labels=[]):
    if len(input_labels) == 0 or len(output_labels) == 0:
        raise RuntimeError("must specify at least one input and one output label")

    print("Downloading input images...")
    input_dir = "./openimages/input/"
    download_images(input_dir, input_labels)
    print("Downloading output iamges...")
    output_dir = "./openimages/output/"
    download_images(output_dir, output_labels)

    options = {
        "validation_split": 0.2,
        "seed": 123,
        "image_size": (256, 256),
        "labels": None,
        "label_mode": None,
        "batch_size": None,
        "shuffle": False,
        "smart_resize": True,
    }

    print("Creating datasets...")
    train_x = image_dataset_from_directory(input_dir, subset="training", **options)
    train_y = image_dataset_from_directory(output_dir, subset="training", **options)
    test_x = image_dataset_from_directory(input_dir, subset="validation", **options)
    test_y = image_dataset_from_directory(output_dir, subset="validation", **options)

    return (
        preprocess_images(train_x, jitter=True),
        preprocess_images(train_y, jitter=True),
        preprocess_images(test_x),
        preprocess_images(test_y),
    )


# https://gist.github.com/devhero/8ae2229d9ea1a59003ced4587c9cb236
def download_tar(tar_url, extract_path="."):
    ftpstream = urllib.request.urlopen(tar_url)
    thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
    thetarfile.extractall(path=extract_path)


def load_cartoons():
    cartoons_dir = "./cartoons"

    print("Downloading cartoons...")
    download_tar(
        "https://storage.cloud.google.com/cartoonset_public_files/cartoonset10k.tgz",
        extract_path=cartoons_dir,
    )

    options = {
        "validation_split": 0.2,
        "seed": 123,
        "image_size": (256, 256),
        "labels": None,
        "label_mode": None,
        "batch_size": None,
        "shuffle": False,
        "smart_resize": True,
    }

    print("Creating datasets...")
    train = image_dataset_from_directory(cartoons_dir, subset="training", **options)
    test = image_dataset_from_directory(cartoons_dir, subset="validation", **options)

    return preprocess_images(train, jitter=True), preprocess_images(test)


def load_cycle_gan_dataset(dataset):
    data, metadata = tfds.load(
        "cycle_gan/" + dataset,
        with_info=True,
        as_supervised=True,
    )

    train_x, train_y = data["trainA"], data["trainB"]
    test_x, test_y = data["testA"], data["testB"]

    return (
        preprocess_images(train_x, jitter=True),
        preprocess_images(train_y, jitter=True),
        preprocess_images(test_x),
        preprocess_images(test_y),
    )
