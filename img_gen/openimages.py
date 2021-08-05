from openimages.download import download_images
from tensorflow.keras.preprocessing import image_dataset_from_directory


def load_images(input_labels=[], output_labels=[]):
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

    return train_x, train_y, test_x, test_y
