import os

from google.cloud import storage

os.environ["GOOGLE_CLOUD_PROJECT"] = "img-gen-319216"


def save_figure(key):
    print("saving figure")
    client = storage.Client()
    bucket = client.get_bucket("img-gen-training")
    blob = bucket.blob(key)
    blob.upload_from_filename(filename="temp.png")
