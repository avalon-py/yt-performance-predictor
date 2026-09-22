"""
Precompute face/text/color features once per thumbnail, cached to
data/visual_features.csv. Incremental -- reruns skip video_ids already
present, same pattern as the main ingestion pipeline.

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

from features.visual_features import detect_faces, detect_text_overlay, color_stats

CSV_PATH = "data/videos.csv"
OUTPUT_PATH = "data/visual_features.csv"


def main():
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
    for i, row in todo.iterrows():
        if not os.path.exists(row["thumbnail_path"]):
            continue
        try:
            image = Image.open(row["thumbnail_path"]).convert("RGB")
            has_face, face_count = detect_faces(image)
            has_text = detect_text_overlay(image)
            colors = color_stats(image)

            rows.append({
                "video_id": row["video_id"],
                "has_face": has_face,
                "face_count": face_count,
                "has_text_overlay": has_text,
                **colors,
            })
        except Exception as e:
            print(f"  [warn] failed on {row['video_id']}: {e}")

        if len(rows) >= CHECKPOINT_EVERY:
            flush(rows)
            rows = []

    flush(rows)
    print("\nDone.")


if __name__ == "__main__":
    main()