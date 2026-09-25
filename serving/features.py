"""
Serving-side feature reconstruction for a single (thumbnail, title, channel)
input. This module is the deliberate second implementation of
build_tabular_matrix()'s per-row math from models/dataset.py -- training
processes a DataFrame of many rows at once, serving processes one request at
a time, so the code can't just be imported and called. Every column and every
concatenation order below must match models/dataset.py exactly, which is what
tests/test_bundle_parity.py checks.

For the encoders (image + title -> embeddings), this module does NOT
reimplement anything: it calls straight into models/precompute_embeddings.py
(build_transform, postprocess, embed_titles_clip) so there is exactly one
source of truth for that preprocessing.
"""

import numpy as np
import pandas as pd
import torch
from PIL import Image

from features.title_features import title_features
from models.precompute_embeddings import postprocess, embed_titles_clip


def encode_image(pil_image, encode_fn, transform, cfg, device):
    """Mirrors embed_images() in precompute_embeddings.py for a single image:
    same transform, same encode_fn, same postprocess (unit-RMS rescale)."""
    x = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        out = encode_fn(x).float().cpu().numpy()
    return postprocess(out, cfg)[0]


def encode_title_clip(title, clip_model, clip_text_model_name, device):
    """Mirrors embed_titles_clip() for a single title. Reuses that function
    directly (via a one-row DataFrame) rather than re-deriving the tokenizer
    call, so a change to padding/truncation there can't silently drift from
    serving."""
    emb = embed_titles_clip(pd.DataFrame({"title": [title]}), clip_model, clip_text_model_name, device)
    return postprocess(emb, {"rescale": True})[0]


def cosine_sim(a, b):
    """Same definition as train.py's cosine_rows, for one pair of vectors.
    Scale-invariant, so it doesn't matter that these embeddings are already
    unit-RMS rescaled by postprocess() above."""
    denom = max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8)
    return float(np.dot(a, b) / denom)


def build_tabular_row(bundle, *, title, subscriber_count_at_upload,
                       trailing_avg_views, duration_seconds, genre, clip_sim=None):
    """Builds one row of the tabular feature vector in the exact column order
    build_tabular_matrix() produces: [log_scaled, numeric_scaled, bool_vals,
    genre_onehot]. Column membership and order come from bundle["feature_columns"],
    a snapshot taken at export time -- not from the live models.dataset module
    lists, which load_data() can mutate (appending "clip_sim") depending on
    USE_SIM. Using the bundle's snapshot is what keeps this correct regardless
    of what any other process has done to those module-level lists since.
    """
    cols = bundle["feature_columns"]
    log_cols, numeric_cols, bool_cols = cols["log_cols"], cols["numeric_cols"], cols["bool_cols"]
    log_scaler, numeric_scaler = bundle["scaler"]

    raw = {
        "subscriber_count_at_upload": subscriber_count_at_upload,
        "trailing_avg_views": trailing_avg_views,
        "duration_seconds": duration_seconds,
        **title_features(title),
    }
    if "clip_sim" in numeric_cols:
        if clip_sim is None:
            raise ValueError(
                "this bundle was trained with USE_SIM=1 (clip_sim is a required "
                "tabular column) but no clip_sim was computed for this request"
            )
        raw["clip_sim"] = clip_sim
    elif clip_sim is not None:
        raise ValueError("clip_sim was provided but this bundle was not trained with USE_SIM=1")

    missing = [c for c in (*log_cols, *numeric_cols, *bool_cols) if c not in raw]
    if missing:
        raise ValueError(f"missing required feature(s) for this bundle: {missing}")

    log_vals = np.array([[raw[c] for c in log_cols]], dtype=float)
    numeric_vals = np.array([[raw[c] for c in numeric_cols]], dtype=float)
    log_scaled = log_scaler.transform(np.log1p(log_vals))
    numeric_scaled = numeric_scaler.transform(numeric_vals)
    bool_vals = np.array([[float(raw[c]) for c in bool_cols]], dtype=np.float32)

    genre_categories = bundle["genre_categories"]
    genre_onehot = np.zeros((1, len(genre_categories)), dtype=np.float32)
    if genre in genre_categories:
        genre_onehot[0, genre_categories.index(genre)] = 1.0
    # else: all-zero, same as build_tabular_matrix's behaviour for a genre
    # the training set never saw.

    tabular = np.concatenate([log_scaled, numeric_scaled, bool_vals, genre_onehot], axis=1)
    return tabular.astype(np.float32)[0]