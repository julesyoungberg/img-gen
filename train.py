GCP_PROJECT = "img-gen-319216"
GCP_BUCKET = "img-gen-training"

import tensorflow_datasets as tfds

from img_gen.cycle_gan import CycleGAN, find_optimal_cycle_gan
from img_gen.img import preprocess_images
from img_gen.data import load_cycle_gan_dataset

train_x, train_y, test_x, test_y = load_cycle_gan_dataset("vangogh2photo")

# build & train the model
print("finding optimal cycle gan")
cyc_gan = find_optimal_cycle_gan(
    train_x,
    train_y,
    test_x,
    test_y,
    checkpoints=True,
    use_cloud=True,
    cloud_project=GCP_PROJECT,
    cloud_bucket=GCP_BUCKET,
    name="vangogh2photo",
)

# print results
cyc_gan.print_losses()
cyc_gan.plot_losses()

print("Test Results")
cyc_gan.generate_images(test_x, test_y)

gen_g_loss, gen_f_loss, dis_x_loss, dis_y_loss = cyc_gan.scores(test_x, test_y)
print("Test Losses")
print(
    f"gen_g: {gen_g_loss}, gen_f: {gen_f_loss}, dis_x: {dis_x_loss}, dis_y: {dis_y_loss}"
)
