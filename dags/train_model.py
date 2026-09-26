"""train_model -- retrain the late-fusion model on all currently embedded rows.

Runs models/train.py inside a ytpp-worker container.

Unlike ingest_new/embed_new, this task's outputs must survive the container's
removal: auto_remove="success" wipes anything written only inside the
container's own filesystem, and train.py writes its checkpoint, model bundle,
results log and plots to local paths (models/checkpoints, models/bundles,
experiments/results.jsonl, models/plots). So this DAG bind-mounts those paths
back to the host. Writing into the same host directory the `api` service
already mounts read-only (./models/bundles) means a freshly trained bundle is
picked up without restarting the API -- just re-point MODEL_BUNDLE_PATH or
restart `api` to load it, depending on how serving/bundle.py picks the file.

Requires HOST_PROJECT_DIR in .env: the path to this repo as the *Docker
daemon* sees it (reached via /var/run/docker.sock), not the path inside the
scheduler container. On Docker Desktop for Windows this is the same path you
use for Docker Desktop's file sharing, e.g.
D:/Material/Programming/Machine Learning/yt-performance-predictor/yt-performance-predictor
Forward slashes, even on Windows.

Before the first run, make sure the target directories exist on the host:
    mkdir -p models/bundles models/plots
(models/checkpoints and experiments already exist in this repo.)
"""
import os

import pendulum
from airflow.sdk import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

WORKER_IMAGE = "ytpp-worker:latest"
NETWORK = "ytpp_net"

# Fail at DAG-parse time with a clear message rather than at task-run time
# with a confusing Docker mount error, if this hasn't been set up yet.
HOST_PROJECT_DIR = os.environ["HOST_PROJECT_DIR"]

WORKER_ENV = {
    "POSTGRES_HOST": "postgres",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USER": os.environ.get("POSTGRES_USER", ""),
    "POSTGRES_PASSWORD": os.environ.get("POSTGRES_PASSWORD", ""),
    "POSTGRES_DB": os.environ.get("POSTGRES_DB", ""),
}

MOUNTS = [
    Mount(source=f"{HOST_PROJECT_DIR}/models/checkpoints", target="/app/models/checkpoints", type="bind"),
    Mount(source=f"{HOST_PROJECT_DIR}/models/bundles", target="/app/models/bundles", type="bind"),
    Mount(source=f"{HOST_PROJECT_DIR}/models/plots", target="/app/models/plots", type="bind"),
    Mount(source=f"{HOST_PROJECT_DIR}/experiments", target="/app/experiments", type="bind"),
]

with DAG(
    dag_id="train_model",
    description="Retrain the late-fusion model on all embedded rows",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["ytpp", "training"],
) as dag:
    run_training = DockerOperator(
        task_id="run_training",
        image=WORKER_IMAGE,
        command=["python", "-m", "models.train"],
        docker_url="unix://var/run/docker.sock",
        network_mode=NETWORK,
        environment=WORKER_ENV,
        mounts=MOUNTS,
        auto_remove="success",
        mount_tmp_dir=False,
        # 200 epochs w/ early stopping (patience 10), CPU-bound -- give it
        # real headroom. Tighten once you know how long a real run takes.
        execution_timeout=pendulum.duration(hours=6),
    )