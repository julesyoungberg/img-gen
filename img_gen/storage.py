import os

from google.cloud import storage

os.environ["GOOGLE_CLOUD_PROJECT"] = "img-gen-319216"

client = storage.Client.from_service_account_json("service_account.json")
bucket = client.get_bucket("img-gen-training")


def save_file(local_filename, bucket_key):
    blob = bucket.blob(bucket_key)
    blob.upload_from_filename(filename=local_filename)


def save_figure(key):
    save_file("temp.png", key)
