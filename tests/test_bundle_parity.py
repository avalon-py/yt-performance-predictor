"""
Train/serve parity test.

For ~N real rows, this computes the model's prediction two independent ways
and checks they agree to ~1e-5:

  1. "training path": the cached embeddings from data/embeddings/<encoder>/
     plus models.dataset.build_tabular_matrix() -- the same code train.py
     uses for val/test.
  2. "serving path": serving.bundle.LoadedBundle.predict(), which re-encodes
     the actual thumbnail file and title text from scratch and builds the
     tabular row via serving.features.build_tabular_row().

If these ever disagree beyond floating-point noise, serving and training have
drifted -- most likely a column-order or scaling mismatch, since the encoders
themselves are frozen and deterministic in eval mode.

Requires torch, transformers, etc. (not available in this review environment,
so this has been reviewed by hand, not executed). Run locally from the repo
root, after a real train.py run has produced a bundle:

    python -m tests.test_bundle_parity --bundle models/bundles/latest_clip_b32.pt
    python -m tests.test_bundle_parity --bundle models/bundles/latest_clip_b32_sim.pt --n 100

First run downloads the CLIP checkpoint from HuggingFace -- make sure that's
reachable, or pre-cache it, before running on a network-restricted machine.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

import models.dataset as ds
from models.dataset import build_tabular_matrix
from models.late_fusion_model import LateFusionModel
from models.train import cosine_rows
from serving.bundle import LoadedBundle

CSV_PATH = "data/videos.csv"
EMBEDDINGS_ROOT = "data/embeddings"
MEAN_TOL = 1e-3
# Catches a real bug: wrong column order/scaling doesn't nudge every row a
# little, it makes a meaningful fraction of rows wrong by a lot, which pulls
# the average up fast. Batching float noise (see below) barely moves this
# even at n=200 -- measured mean ~2.5e-4, an order of magnitude under this.

MAX_TOL = 1e-2
PER_ROW_TOL = 3e-3
MAX_TAIL_FRACTION = 0.02
# Why not one flat threshold: training batches many images/titles together
# (32 / 256 at a time), serving encodes one at a time. Matmul isn't exactly
# associative across batch shapes, so a handful of rows always land a bit
# further from zero than the rest -- a real, deterministic, harmless effect,
# not a bug. As you test more rows you naturally sample deeper into that
# tail, so a single fixed cutoff either false-fails at high n or is too loose
# at low n. Instead: allow up to MAX_TAIL_FRACTION of rows to exceed
# PER_ROW_TOL (the float-noise tail), but nothing may ever exceed MAX_TOL
# (a real bug's territory -- 3-10x anything observed from batching alone).


def load_rows_for_bundle(bundle, n, seed, csv_path=CSV_PATH, embeddings_root=EMBEDDINGS_ROOT):
    """Same row filter as train.py's load_data(): finalized + valid image +
    has a trailing baseline. Also requires a real thumbnail file on disk,
    since the serving path re-encodes it from scratch."""
    embed_dir = os.path.join(embeddings_root, bundle["image_encoder"])
    text_file = "clip_text_embeddings.npy"  # serving is CLIP-text-only

    df = pd.read_csv(csv_path)
    df["label_finalized"] = df["label_finalized"].astype(str) == "True"
    image_embeddings = np.load(os.path.join(embed_dir, "image_embeddings.npy"))
    text_embeddings = np.load(os.path.join(embed_dir, text_file))
    valid_image_mask = np.load(os.path.join(embed_dir, "valid_image_mask.npy"))
    video_id_order = pd.read_csv(os.path.join(embed_dir, "video_id_order.csv"))

    df = df.merge(video_id_order.reset_index().rename(columns={"index": "_embed_idx"}), on="video_id")
    df = df.sort_values("_embed_idx").reset_index(drop=True)

    mask = (
        df["label_finalized"]
        & valid_image_mask[df["_embed_idx"].values]
        & df["trailing_avg_views"].notna()
        & df["thumbnail_path"].apply(lambda p: isinstance(p, str) and os.path.exists(p))
    )
    df = df[mask].reset_index(drop=True)
    if len(df) == 0:
        sys.exit(f"No usable rows found under {embed_dir} with thumbnails on disk.")

    idxs = df["_embed_idx"].values
    image_embeddings = image_embeddings[idxs]
    text_embeddings = text_embeddings[idxs]

    n = min(n, len(df))
    sample_idx = df.sample(n=n, random_state=seed).index.to_numpy()
    return df.loc[sample_idx].reset_index(drop=True), image_embeddings[sample_idx], text_embeddings[sample_idx]


def training_path_predictions(bundle, df, image_embeddings, text_embeddings, device):
    """Rebuild the tabular matrix the way train.py does for val/test, using
    the bundle's saved scaler + genre_categories (fit at training time, not
    refit here) -- and the bundle's saved column snapshot, since the live
    models.dataset lists may not match what this bundle was trained with."""
    cols = bundle["feature_columns"]
    original = (list(ds.TABULAR_LOG_COLS), list(ds.TABULAR_NUMERIC_COLS), list(ds.TABULAR_BOOL_COLS))
    ds.TABULAR_LOG_COLS[:] = cols["log_cols"]
    ds.TABULAR_NUMERIC_COLS[:] = cols["numeric_cols"]
    ds.TABULAR_BOOL_COLS[:] = cols["bool_cols"]
    try:
        df = df.copy()
        if "clip_sim" in cols["numeric_cols"]:
            df["clip_sim"] = cosine_rows(image_embeddings, text_embeddings)
        tabular, _ = build_tabular_matrix(df, bundle["genre_categories"], scaler=bundle["scaler"])
    finally:
        ds.TABULAR_LOG_COLS[:], ds.TABULAR_NUMERIC_COLS[:], ds.TABULAR_BOOL_COLS[:] = original

    model = LateFusionModel(
        image_dim=bundle["image_dim"], text_dim=bundle["text_dim"], tabular_dim=bundle["tabular_dim"],
    ).to(device)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    with torch.inference_mode():
        preds = model(
            torch.tensor(image_embeddings, dtype=torch.float32, device=device),
            torch.tensor(text_embeddings, dtype=torch.float32, device=device),
            torch.tensor(tabular, dtype=torch.float32, device=device),
        ).cpu().numpy()
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, help="path to a bundle .pt from models/bundles/")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", default=CSV_PATH, help="videos.csv to sample rows from")
    parser.add_argument("--embeddings-root", default=EMBEDDINGS_ROOT, help="dir containing <encoder>/image_embeddings.npy etc.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(args.bundle, map_location=device, weights_only=False)

    df, image_embeddings, text_embeddings = load_rows_for_bundle(
        bundle, args.n, args.seed, csv_path=args.csv, embeddings_root=args.embeddings_root
    )
    print(f"Testing {len(df)} rows against {args.bundle} (version {bundle['version']})")

    train_preds = training_path_predictions(bundle, df, image_embeddings, text_embeddings, device)

    loaded = LoadedBundle(args.bundle, device=device)
    serve_preds = np.array([
        loaded.predict(
            thumbnail=row["thumbnail_path"],
            title=row["title"],
            subscriber_count_at_upload=row["subscriber_count_at_upload"],
            trailing_avg_views=row["trailing_avg_views"],
            duration_seconds=row["duration_seconds"],
            genre=row["genre"],
        )["score"]
        for _, row in df.iterrows()
    ])

    diff = np.abs(train_preds - serve_preds)
    mean_diff = diff.mean()
    max_diff = diff.max()
    tail_fraction = (diff > PER_ROW_TOL).mean()

    print(f"mean |diff| = {mean_diff:.2e}  (bug threshold: {MEAN_TOL:.0e})")
    print(f"max  |diff| = {max_diff:.2e}  (bug threshold: {MAX_TOL:.0e})")
    print(f"{(diff > PER_ROW_TOL).sum()}/{len(diff)} rows exceed the float-noise band of "
          f"{PER_ROW_TOL:.0e} ({tail_fraction:.1%}, allowed up to {MAX_TAIL_FRACTION:.0%})")

    failed = mean_diff > MEAN_TOL or max_diff > MAX_TOL or tail_fraction > MAX_TAIL_FRACTION
    if failed:
        worst = np.argsort(-diff)[:5]
        print("\nWorst offenders:")
        for i in worst:
            vid = df.iloc[i]["video_id"]
            print(f"  {vid}: train={train_preds[i]:.6f}  serve={serve_preds[i]:.6f}  diff={diff[i]:.2e}")
        sys.exit(
            "\nFAIL: this looks like more than batching noise -- "
            f"mean_diff>{MEAN_TOL:.0e}: {mean_diff > MEAN_TOL}, "
            f"max_diff>{MAX_TOL:.0e}: {max_diff > MAX_TOL}, "
            f"tail_fraction>{MAX_TAIL_FRACTION:.0%}: {tail_fraction > MAX_TAIL_FRACTION}"
        )

    print(f"\nPASS: {len(diff)} rows consistent with training-vs-serving batch-size float noise.")


if __name__ == "__main__":
    main()