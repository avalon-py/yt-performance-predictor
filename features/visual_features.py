"""
Face presence/count, text overlay detection, and color distribution stats
for a thumbnail image. Each function takes a PIL Image and returns primitive
values -- no caching logic here, that lives in precompute_visual_features.py.
"""

import os
import numpy as np
import torch
from PIL import Image

_face_detector = None
_ocr_reader = None

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OCR_MODEL_DIR = os.path.join(os.getcwd(), ".cache", "easyocr", "model")
OCR_USER_NETWORK_DIR = os.path.join(os.getcwd(), ".cache", "easyocr", "user_network")


def _get_face_detector():
    global _face_detector
    if _face_detector is None:
        from facenet_pytorch import MTCNN
        _face_detector = MTCNN(keep_all=True, device=_DEVICE)
    return _face_detector


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        os.makedirs(OCR_MODEL_DIR, exist_ok=True)
        os.makedirs(OCR_USER_NETWORK_DIR, exist_ok=True)
        _ocr_reader = easyocr.Reader(
            ["en"], gpu=(_DEVICE == "cuda"),
            model_storage_directory=OCR_MODEL_DIR,
            user_network_directory=OCR_USER_NETWORK_DIR,
        )
    return _ocr_reader


def detect_faces(image: Image.Image):
    """Returns (has_face: bool, face_count: int). Single-image path, kept for
    compatibility / fallback -- prefer detect_faces_batch for bulk processing."""
    detector = _get_face_detector()
    boxes, _ = detector.detect(image)
    if boxes is None:
        return False, 0
    return True, len(boxes)


def detect_faces_batch(images: list[Image.Image], batch_size: int = 32, resize_to: tuple[int, int] = (640, 360)):
    """Returns a list of (has_face, face_count) tuples, one per image, in
    input order -- computed via batched MTCNN.detect() calls.

    MTCNN's batched path stacks inputs into a single tensor, which requires
    every image in the batch to share the same H x W. Real thumbnails don't
    (different source resolutions, aspect ratios, re-encodes), so we resize
    a copy of each image to `resize_to` before detection. This only affects
    box coordinates, which we discard anyway -- we just need has_face /
    face_count, and resizing doesn't change whether a face is detected in
    any way that matters here.

    Falls back to per-image detection (on the *original* images) if a batch
    still fails for some other reason, so one bad file doesn't drop the batch.
    """
    detector = _get_face_detector()
    results = [None] * len(images)

    for start in range(0, len(images), batch_size):
        chunk = images[start:start + batch_size]
        resized = [img.resize(resize_to) for img in chunk]
        try:
            boxes_list, _ = detector.detect(resized)
            for offset, boxes in enumerate(boxes_list):
                results[start + offset] = (False, 0) if boxes is None else (True, len(boxes))
        except Exception as e:
            print(f"  [warn] batched face detection failed ({e}), falling back to per-image for this chunk")
            for offset, img in enumerate(chunk):
                results[start + offset] = detect_faces(img)

    return results

def detect_text_overlay(image: Image.Image):
    """Returns True if any text region is detected.

    Uses detect() rather than readtext() -- we only need presence, not the
    actual transcribed text, and skipping the recognition step (the slow
    part of full OCR) is a large speedup for a binary flag like this.
    Not batched: EasyOCR's detect() doesn't accept a list of images cleanly.
    """
    reader = _get_ocr_reader()
    horizontal_boxes, free_boxes = reader.detect(np.array(image))
    return bool(horizontal_boxes and horizontal_boxes[0]) or bool(free_boxes and free_boxes[0])


def color_stats(image: Image.Image):
    """Pure numpy, no model needed. Returns mean saturation, mean brightness,
    brightness std (contrast proxy), and warm-hue pixel ratio (psychologically
    linked to urgency/excitement in thumbnail-design practice)."""
    hsv = np.array(image.convert("HSV")).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    warm_mask = (h <= 42) | (h >= 217)

    return {
        "mean_saturation": float(s.mean()),
        "mean_brightness": float(v.mean()),
        "brightness_std": float(v.std()),
        "warm_hue_ratio": float(warm_mask.mean()),
    }