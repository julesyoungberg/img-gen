import os

from google.cloud import storage

os.environ["GOOGLE_CLOUD_PROJECT"] = "img-gen-319216"


def save_figure(key):
    client = storage.Client.from_service_account_json("service_account.json")
    bucket = client.get_bucket("img-gen-training")
    blob = bucket.blob(key)
    blob.upload_from_filename(filename="temp.png")
