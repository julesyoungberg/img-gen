from img_gen.openimages import load_images

train_x, train_y, test_x, test_y = load_images(
    input_labels=["Cat"], output_labels=["Dog"]
)

print("train_x: ", train_x.shape)
print("train_y: ", train_y.shape)
print("test_x: ", test_x.shape)
print("test_y: ", test_y.shape)
