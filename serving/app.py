"""
FastAPI wrapper around serving.bundle.LoadedBundle.

Loads the model bundle once at process startup (lifespan, not the deprecated
on_event hooks) and serves it from memory for every request -- no
per-request reload.

Run locally:
    uvicorn serving.app:app --host 0.0.0.0 --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from serving.bundle import LoadedBundle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("serving")

BUNDLE_PATH = os.environ.get("MODEL_BUNDLE_PATH", "models/bundles/latest_clip_b32_clip.pt")

_bundle_holder = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model bundle from %s", BUNDLE_PATH)
    _bundle_holder["bundle"] = LoadedBundle(BUNDLE_PATH)
    logger.info(
        "Bundle loaded: version=%s train_end=%s",
        _bundle_holder["bundle"].bundle["version"],
        _bundle_holder["bundle"].bundle["train_end"],
    )
    yield
    _bundle_holder.clear()


app = FastAPI(title="yt-performance-predictor", lifespan=lifespan)


def get_bundle() -> LoadedBundle:
    bundle = _bundle_holder.get("bundle")
    if bundle is None:
        # Shouldn't happen outside of tests that bypass lifespan.
        raise HTTPException(status_code=503, detail="model not loaded")
    return bundle


@app.get("/health")
def health():
    bundle = get_bundle()
    return {
        "status": "ok",
        "model_version": bundle.bundle["version"],
        "train_end": bundle.bundle["train_end"],
    }


@app.post("/predict")
async def predict(
    thumbnail: UploadFile = File(...),
    title: str = Form(...),
    subscriber_count_at_upload: float = Form(...),
    trailing_avg_views: float = Form(...),
    duration_seconds: float = Form(...),
    genre: str = Form(...),
):
    bundle = get_bundle()

    raw = await thumbnail.read()
    try:
        image = Image.open(__import__("io").BytesIO(raw))
        image.load()  # force decode now, not lazily inside bundle.predict()
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="thumbnail is not a readable image")

    try:
        result = bundle.predict(
            thumbnail=image,
            title=title,
            subscriber_count_at_upload=subscriber_count_at_upload,
            trailing_avg_views=trailing_avg_views,
            duration_seconds=duration_seconds,
            genre=genre,
        )
    except ValueError as e:
        # build_tabular_row raises ValueError for bundle/request mismatches
        # (missing clip_sim, unexpected clip_sim, missing feature columns).
        raise HTTPException(status_code=400, detail=str(e))

    return result