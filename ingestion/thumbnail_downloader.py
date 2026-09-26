"""Downloads and caches thumbnail images in MinIO, by video_id."""

import requests
from botocore.exceptions import ClientError

BUCKET_NAME = "thumbnails"


def ensure_bucket_exists(client, bucket_name=BUCKET_NAME):
    try:
        client.head_bucket(Bucket=bucket_name)
    except ClientError:
        client.create_bucket(Bucket=bucket_name)


def download_thumbnail(video_id, thumbnails, minio_client, bucket_name=BUCKET_NAME):
    for quality in ("maxres", "standard", "high", "medium", "default"):
        if quality in thumbnails:
            url = thumbnails[quality]["url"]
            break
    else:
        return None

    object_key = f"{video_id}.jpg"

    try:
        minio_client.head_object(Bucket=bucket_name, Key=object_key)
        return object_key  # already uploaded, skip re-fetching
    except ClientError:
        pass  # doesn't exist yet -- fetch it below

    resp = requests.get(url)
    if resp.status_code == 200:
        minio_client.put_object(
            Bucket=bucket_name, Key=object_key, Body=resp.content, ContentType="image/jpeg"
        )
        return object_key
    return None