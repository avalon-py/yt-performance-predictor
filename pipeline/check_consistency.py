"""
Reconciles videos.thumbnail_path (Postgres) against actual objects in the
MinIO thumbnails bucket. Read-only -- just reports mismatches, changes nothing.

Usage:
    python -m pipeline.check_consistency
"""

import os

import boto3
import pandas as pd
from botocore.config import Config
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine

DB_URL = (
    f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:"
    f"{os.environ['POSTGRES_PASSWORD']}@localhost:5432/{os.environ['POSTGRES_DB']}"
)
engine = create_engine(DB_URL)

MINIO_BUCKET = "thumbnails"
minio_client = boto3.client(
    "s3",
    endpoint_url=f"http://{os.environ['MINIO_ENDPOINT']}",
    aws_access_key_id=os.environ["MINIO_ROOT_USER"],
    aws_secret_access_key=os.environ["MINIO_ROOT_PASSWORD"],
    config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
)


def list_minio_object_keys(bucket_name):
    keys = set()
    paginator = minio_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name):
        for obj in page.get("Contents", []):
            keys.add(obj["Key"])
    return keys


def main():
    df = pd.read_sql("SELECT video_id, thumbnail_path FROM videos WHERE thumbnail_path IS NOT NULL", engine)
    postgres_keys = set(df["thumbnail_path"])
    print(f"Postgres references {len(postgres_keys)} thumbnail objects")

    minio_keys = list_minio_object_keys(MINIO_BUCKET)
    print(f"MinIO bucket '{MINIO_BUCKET}' actually contains {len(minio_keys)} objects")

    missing_in_minio = postgres_keys - minio_keys
    orphaned_in_minio = minio_keys - postgres_keys

    print(f"\nReferenced in Postgres but MISSING from MinIO: {len(missing_in_minio)}")
    if missing_in_minio:
        print(f"  e.g. {list(missing_in_minio)[:5]}")

    print(f"Present in MinIO but not referenced by any row: {len(orphaned_in_minio)}")
    if orphaned_in_minio:
        print(f"  e.g. {list(orphaned_in_minio)[:5]}")

    if not missing_in_minio and not orphaned_in_minio:
        print("\nFully consistent -- every referenced thumbnail exists, no orphans.")


if __name__ == "__main__":
    main()