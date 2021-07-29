from google.cloud import storage
import matplotlib.pyplot as plt


def save_figure(fig, key):
    print("saving figure")
    plt.savefig("temp.png", dpi=300, bbox_inches="tight")
    client = storage.Client(project="img-gen-319216")
    bucket = client.get_bucket("img-gen-training")
    blob = bucket.blob(key)
    blob.upload_from_filename(filename="temp.png")
