"""embed_new -- precompute CLIP embeddings for rows ingest_new added.

Runs models/precompute_embeddings.py, which is idempotent by design (only
processes rows WHERE image_embedding IS NULL OR text_embedding IS NULL), so
this is also safe to trigger manually or re-run after a failure.

schedule=None: this DAG is only ever started by ingest_new's
trigger_embed_new task, not on its own clock. Trigger it manually
(dags trigger embed_new) to backfill without a fresh ingestion run.
"""
import os

import pendulum
from airflow.sdk import DAG
from airflow.providers.docker.operators.docker import DockerOperator

WORKER_IMAGE = "ytpp-worker:latest"
NETWORK = "ytpp_net"

WORKER_ENV = {
    "POSTGRES_HOST": "postgres",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USER": os.environ.get("POSTGRES_USER", ""),
    "POSTGRES_PASSWORD": os.environ.get("POSTGRES_PASSWORD", ""),
    "POSTGRES_DB": os.environ.get("POSTGRES_DB", ""),
    "MINIO_ENDPOINT": "minio:9000",
    "MINIO_ROOT_USER": os.environ.get("MINIO_ROOT_USER", ""),
    "MINIO_ROOT_PASSWORD": os.environ.get("MINIO_ROOT_PASSWORD", ""),
}

with DAG(
    dag_id="embed_new",
    description="Precompute CLIP image/text embeddings for un-embedded rows",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["ytpp", "embeddings"],
) as dag:
    run_embeddings = DockerOperator(
        task_id="run_embeddings",
        image=WORKER_IMAGE,
        command=["python", "-m", "models.precompute_embeddings"],
        docker_url="unix://var/run/docker.sock",
        network_mode=NETWORK,
        environment=WORKER_ENV,
        auto_remove="success",
        mount_tmp_dir=False,
        execution_timeout=pendulum.duration(hours=3),
    )
