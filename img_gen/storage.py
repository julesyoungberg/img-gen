import os

from google.cloud import storage


def get_bucket(project="img-gen-319216", bucket="img-gen-training"):
    os.environ["GOOGLE_CLOUD_PROJECT"] = project
    client = storage.Client.from_service_account_json("service_account.json")
    bucket = client.get_bucket(bucket)
    return bucket


def save_file(
    local_filename, bucket_key, project="img-gen-319216", bucket="img-gen-training"
):
    bucket = get_bucket(project=project, bucket=bucket)
    blob = bucket.blob(bucket_key)
    blob.upload_from_filename(filename=local_filename)


def save_figure(key, project="img-gen-319216", bucket="img-gen-training"):
    save_file("temp.png", key, project=project, bucket=bucket)
