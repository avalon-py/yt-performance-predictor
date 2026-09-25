"""
Loads one exported bundle (see export_bundle() in models/train.py) and
predicts on a single (thumbnail, title, channel) input.

Serving is CLIP-only by design (see README: "Ablations" / roadmap step 1 --
DINOv2 and MiniLM aren't shipped). A bundle trained with a different image or
text encoder is rejected at load time rather than silently mishandled.

Usage:
    bundle = LoadedBundle("models/bundles/latest_clip_b32_clip.pt")
    result = bundle.predict(
        thumbnail="data/images/abc123.jpg",   # path, bytes, or PIL.Image
        title="I Tried This For 30 Days",
        subscriber_count_at_upload=482_000,
        trailing_avg_views=310_000,
        duration_seconds=612,
        genre="Entertainment",
    )
    # -> {"score": ..., "expected_views": ..., "model_version": ..., "train_end": ...}

Not run in this environment (no torch here) -- reviewed by hand against
models/precompute_embeddings.py and models/late_fusion_model.py.
"""

import io

import numpy as np
import torch
from PIL import Image

from features.target import invert_target
from models.late_fusion_model import LateFusionModel
from models.precompute_embeddings import ENCODERS, build_transform, build_image_encoder, load_clip_text_tower
from serving.features import encode_image, encode_title_clip, cosine_sim, build_tabular_row


class LoadedBundle:
    def __init__(self, bundle_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bundle = torch.load(bundle_path, map_location=self.device, weights_only=False)

        image_encoder = self.bundle["image_encoder"]
        if not image_encoder.startswith("clip"):
            raise ValueError(
                f"serving only supports CLIP image encoders, this bundle used {image_encoder!r}. "
                "Retrain with IMAGE_ENCODER=clip_b32 (or clip_b16) before exporting a serving bundle."
            )
        if self.bundle["text_encoder"] != "clip":
            raise ValueError(
                f"serving only supports the CLIP text encoder, this bundle used {self.bundle['text_encoder']!r}. "
                "Retrain with TEXT_ENCODER=clip before exporting a serving bundle."
            )

        self.cfg = ENCODERS[image_encoder]
        self.transform = build_transform(self.cfg, self.bundle["image_mode"])
        self.encode_image_fn, clip_model = build_image_encoder(self.cfg, self.device)

        clip_text_model = self.bundle["clip_text_model"]
        if clip_text_model == self.cfg["hf_name"]:
            # Same checkpoint already loaded above for the vision tower --
            # CLIPModel carries both towers, so reuse it (mirrors
            # precompute_embeddings.py's "if clip_model is not None" branch).
            self.text_tower, self.text_tower_name = clip_model, self.cfg["hf_name"]
        else:
            self.text_tower, self.text_tower_name = load_clip_text_tower(clip_text_model, self.device), clip_text_model

        self.model = LateFusionModel(
            image_dim=self.bundle["image_dim"],
            text_dim=self.bundle["text_dim"],
            tabular_dim=self.bundle["tabular_dim"],
        ).to(self.device)
        self.model.load_state_dict(self.bundle["model_state_dict"])
        self.model.eval()

    def _load_image(self, thumbnail):
        if isinstance(thumbnail, Image.Image):
            return thumbnail
        if isinstance(thumbnail, (bytes, bytearray)):
            return Image.open(io.BytesIO(thumbnail))
        return Image.open(thumbnail)  # path-like

    @torch.inference_mode()
    def predict(self, *, thumbnail, title, subscriber_count_at_upload,
                trailing_avg_views, duration_seconds, genre):
        image = self._load_image(thumbnail)
        image_emb = encode_image(image, self.encode_image_fn, self.transform, self.cfg, self.device)
        text_emb = encode_title_clip(title, self.text_tower, self.text_tower_name, self.device)

        clip_sim = cosine_sim(image_emb, text_emb) if self.bundle["use_sim"] else None

        tabular = build_tabular_row(
            self.bundle, title=title,
            subscriber_count_at_upload=subscriber_count_at_upload,
            trailing_avg_views=trailing_avg_views,
            duration_seconds=duration_seconds,
            genre=genre, clip_sim=clip_sim,
        )

        score = self.model(
            torch.tensor(image_emb, dtype=torch.float32, device=self.device).unsqueeze(0),
            torch.tensor(text_emb, dtype=torch.float32, device=self.device).unsqueeze(0),
            torch.tensor(tabular, dtype=torch.float32, device=self.device).unsqueeze(0),
        ).item()

        return {
            "score": score,
            "expected_views": float(invert_target(score, trailing_avg_views)),
            "model_version": self.bundle["version"],
            "train_end": self.bundle["train_end"],
        }