"""
Precompute face/text/color features once per thumbnail, cached to
data/visual_features.csv. Incremental -- reruns skip video_ids already
present, same pattern as the main ingestion pipeline.

Face detection runs batched (BATCH_SIZE images per MTCNN call); text overlay
and color stats stay per-image since EasyOCR doesn't batch cleanly here.

NOTE: not executed in the environment that generated this file -- no
network access here to download the MTCNN/EasyOCR pretrained weights.
Run on a small subset first before pointing it at the full dataset, same
advice as everywhere else in this project.

Usage:
    python -m models.precompute_visual_features
"""

import os
import pandas as pd
from PIL import Image

from features.visual_features import _DEVICE, detect_faces_batch, detect_text_overlay, color_stats

CSV_PATH = "data/videos.csv"
OUTPUT_PATH = "data/visual_features.csv"
BATCH_SIZE = 32


def process_batch(video_ids, images):
    """Runs batched face detection, then per-image text/color, for one batch.
    Returns a list of row dicts."""
    face_results = detect_faces_batch(images)

    out = []
    for video_id, image, (has_face, face_count) in zip(video_ids, images, face_results):
        try:
            has_text = detect_text_overlay(image)
            colors = color_stats(image)
            out.append({
                "video_id": video_id,
                "has_face": has_face,
                "face_count": face_count,
                "has_text_overlay": has_text,
                **colors,
            })
        except Exception as e:
            print(f"  [warn] failed on {video_id}: {e}")
    return out


def main():
    print(f"Using device: {_DEVICE}")

    df = pd.read_csv(CSV_PATH)
    df = df[df["thumbnail_path"].notna()].reset_index(drop=True)

    already_done = set()
    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH)
        already_done = set(existing["video_id"])
        print(f"Loaded {len(already_done)} already-processed video IDs, skipping those")

    todo = df[~df["video_id"].isin(already_done)]
    print(f"Processing {len(todo)} new thumbnails...")

    CHECKPOINT_EVERY = 200  # write to disk periodically -- don't lose progress if interrupted

    def flush(rows):
        if not rows:
            return
        new_df = pd.DataFrame(rows)
        if os.path.exists(OUTPUT_PATH):
            combined = pd.concat([pd.read_csv(OUTPUT_PATH), new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(OUTPUT_PATH, index=False)
        print(f"  [checkpoint] saved, {len(combined)} total rows in {OUTPUT_PATH}")

    rows = []
    batch_ids, batch_images = [], []

    def flush_batch():
        nonlocal rows, batch_ids, batch_images
        if not batch_images:
            return
        rows.extend(process_batch(batch_ids, batch_images))
        batch_ids, batch_images = [], []

    for i, row in todo.iterrows():
        if not os.path.exists(row["thumbnail_path"]):
            continue
        try:
            image = Image.open(row["thumbnail_path"]).convert("RGB")
        except Exception as e:
            print(f"  [warn] failed to open {row['video_id']}: {e}")
            continue

        batch_ids.append(row["video_id"])
        batch_images.append(image)

        if len(batch_images) >= BATCH_SIZE:
            flush_batch()

        if len(rows) >= CHECKPOINT_EVERY:
            flush(rows)
            rows = []

    flush_batch()
    flush(rows)
    print("\nDone.")


if __name__ == "__main__":
    main()