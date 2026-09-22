"""
Face presence/count, text overlay detection, and color distribution stats
for a thumbnail image. Each function takes a PIL Image and returns primitive
values -- no caching logic here, that lives in precompute_visual_features.py.
"""

import numpy as np
from PIL import Image

_face_detector = None
_ocr_reader = None


def _get_face_detector():
    global _face_detector
    if _face_detector is None:
        from facenet_pytorch import MTCNN
        _face_detector = MTCNN(keep_all=True, device="cpu")
    return _face_detector


import os

OCR_MODEL_DIR = os.path.join(os.getcwd(), ".cache", "easyocr", "model")
OCR_USER_NETWORK_DIR = os.path.join(os.getcwd(), ".cache", "easyocr", "user_network")

def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        os.makedirs(OCR_MODEL_DIR, exist_ok=True)
        os.makedirs(OCR_USER_NETWORK_DIR, exist_ok=True)
        _ocr_reader = easyocr.Reader(
            ["en"], gpu=False,
            model_storage_directory=OCR_MODEL_DIR,
            user_network_directory=OCR_USER_NETWORK_DIR,
        )
    return _ocr_reader


def detect_faces(image: Image.Image):
    """Returns (has_face: bool, face_count: int)."""
    detector = _get_face_detector()
    boxes, _ = detector.detect(image)
    if boxes is None:
        return False, 0
    return True, len(boxes)


def detect_text_overlay(image: Image.Image):
    """Returns True if any text region is detected.

    Uses detect() rather than readtext() -- we only need presence, not the
    actual transcribed text, and skipping the recognition step (the slow
    part of full OCR) is a large speedup for a binary flag like this.
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

    # Warm hues: red/orange/yellow. PIL's HSV hue channel is 0-255, not 0-360;
    # roughly maps red/orange/yellow to ~0-42 and ~217-255 (wraps around red).
    warm_mask = (h <= 42) | (h >= 217)

    return {
        "mean_saturation": float(s.mean()),
        "mean_brightness": float(v.mean()),
        "brightness_std": float(v.std()),
        "warm_hue_ratio": float(warm_mask.mean()),
    }