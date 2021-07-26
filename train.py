# import tensorflow_cloud as tfc

GCP_BUCKET = "img-gen-training"

# tfc.run(
#     requirements_txt="requirements.txt",
#     stream_logs=True,
#     worker_count=0,
#     # https://github.com/tensorflow/cloud/blob/d509c231d6b2efec34a0af5da0ee02535a1f746d/src/python/tensorflow_cloud/core/machine_config.py#L116
#     chief_config=tfc.COMMON_MACHINE_CONFIGS["K80_8X"],
# )

import tensorflow_datasets as tfds

from img_gen.cycle_gan import CycleGAN, find_optimal_cycle_gan
from img_gen.img import preprocess_images

# load the data set
print("loading data")
data, metadata = tfds.load(
    "cycle_gan/vangogh2photo",
    with_info=True,
    as_supervised=True,
)

train_x, train_y = data["trainA"], data["trainB"]
test_x, test_y = data["testA"], data["testB"]

# preprocess data
print("preprocessing data")
train_x = preprocess_images(train_x, jitter=True)
train_y = preprocess_images(train_y, jitter=True)
test_x = preprocess_images(test_x)
test_y = preprocess_images(test_y)

# build & train the model
print("finding optimal cycle gan")
# cyc_gan = find_optimal_cycle_gan(
#     train_x,
#     train_y,
#     test_x,
#     test_y,
#     checkpoints=True,
#     use_cloud=True,
#     cloud_bucket=GCP_BUCKET,
# )
cyc_gan = CycleGAN(
    show_images=False,
    use_cloud=True,
    save_images=True,
    save_models=True,
    name="vangogh2photo",
    cloud_bucket=GCP_BUCKET,
)
cyc_gan.train(train_x, train_y, test_x, test_y, checkpoints=True)

# print results
# cyc_gan.print_losses()
# cyc_gan.plot_losses()

# print("Test Results")
# cyc_gan.generate_images(test_x, test_y)

# gen_g_loss, gen_f_loss, dis_x_loss, dis_y_loss = cyc_gan.scores(test_x, test_y)
# print("Test Losses")
# print(
#     f"gen_g: {gen_g_loss}, gen_f: {gen_f_loss}, dis_x: {dis_x_loss}, dis_y: {dis_y_loss}"
# )
