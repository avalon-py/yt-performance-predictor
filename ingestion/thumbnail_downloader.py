"""Downloads and caches thumbnail images by video_id."""

import os
import requests


def download_thumbnail(video_id, thumbnails, images_dir):
    for quality in ("maxres", "standard", "high", "medium", "default"):
        if quality in thumbnails:
            url = thumbnails[quality]["url"]
            break
    else:
        return None

    path = os.path.join(images_dir, f"{video_id}.jpg")
    if os.path.exists(path):
        return path  # already downloaded, skip re-fetching

    resp = requests.get(url)
    if resp.status_code == 200:
        with open(path, "wb") as f:
            f.write(resp.content)
        return path
    return None