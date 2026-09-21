"""
Orchestrates the full ingestion run: fetch -> download -> derive features -> write CSV.

Usage:
    export YOUTUBE_API_KEY="your_key_here"
    python -m pipeline.run_ingestion
"""

import os
import csv
from datetime import datetime, timezone

import pandas as pd

from ingestion.youtube_client import get_channel_info, get_video_ids, get_video_details, get_category_name
from ingestion.thumbnail_downloader import download_thumbnail
from ingestion.tubecensus_client import get_subscriber_count_at
from features.title_features import parse_duration_iso8601_to_seconds, title_features
from features.trailing_views import compute_trailing_views

# Fill with channel handles (e.g. "@mkbhd") or raw channel IDs (UC...).
CHANNELS = [
    "@example_channel_1",
    "@example_channel_2",
    # ... your list of 100 goes here
]

PUBLISHED_AFTER = "2025-01-01T00:00:00Z"
LABEL_MATURITY_DAYS = 28
OUTPUT_DIR = "data"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
CSV_PATH = os.path.join(OUTPUT_DIR, "videos.csv")


def process_channel(channel_ref, rows):
    print(f"Processing {channel_ref}...")
    info = get_channel_info(channel_ref)
    if info is None:
        return
    channel_id, uploads_playlist_id, subscriber_count = info

    video_ids = get_video_ids(uploads_playlist_id, PUBLISHED_AFTER)
    print(f"  found {len(video_ids)} videos since {PUBLISHED_AFTER}")

    details = get_video_details(video_ids)
    now = datetime.now(timezone.utc)

    for item in details:
        video_id = item["id"]
        snippet = item["snippet"]
        stats = item.get("statistics", {})
        content = item["contentDetails"]

        published_at = snippet["publishedAt"]
        pub_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        age_days = (now - pub_dt).days

        duration_seconds = parse_duration_iso8601_to_seconds(content["duration"])
        views = int(stats.get("viewCount", 0))
        genre = get_category_name(snippet.get("categoryId"))

        thumb_path = download_thumbnail(video_id, snippet["thumbnails"], IMAGES_DIR)

        sub_count_at_upload = get_subscriber_count_at(
            channel_id, published_at, fallback_count=subscriber_count
        )

        rows.append({
            "video_id": video_id,
            "channel_id": channel_id,
            "channel_ref": channel_ref,
            "title": snippet["title"],
            "published_at": published_at,
            "duration_seconds": duration_seconds,
            "is_short": duration_seconds <= 180,
            "views": views,
            "label_finalized": age_days >= LABEL_MATURITY_DAYS,
            "subscriber_count_at_upload": sub_count_at_upload,
            "genre": genre,
            "thumbnail_path": thumb_path,
            **title_features(snippet["title"]),
        })


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    rows = []
    for channel_ref in CHANNELS:
        try:
            process_channel(channel_ref, rows)
        except Exception as e:
            print(f"  [error] {channel_ref}: {e}")

    df = pd.DataFrame(rows)
    df = compute_trailing_views(df)
    df.to_csv(CSV_PATH, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"\nDone. {len(df)} rows written to {CSV_PATH}")
    print(f"Images saved to {IMAGES_DIR}/")


if __name__ == "__main__":
    main()