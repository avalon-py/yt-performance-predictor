"""
Precompute frozen image and title embeddings ONCE, cache to disk. v1 doesn't
fine-tune either encoder, so re-running them every training epoch would be
pure waste -- this script runs the expensive forward passes exactly once.

NOTE: not executed/tested in the environment that generated this file --
no GPU/network access to download pretrained weights here. Confirm output
shapes with a small subset before running on your full dataset.

Usage:
    python -m models.precompute_embeddings
"""

import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from sentence_transformers import SentenceTransformer

CSV_PATH = "data/videos.csv"
EMBEDDINGS_DIR = "data/embeddings"
IMAGE_EMBED_DIM = 384  # DINOv2 ViT-S/14's output dim (no "tiny" variant exists -- S is smallest)
TEXT_MODEL_NAME = "all-MiniLM-L6-v2"

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_image_encoder():
    # Self-supervised features (no ImageNet-category bottleneck) -- a better fit
    # than a supervised classifier backbone for a "will this get clicked" task.
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False  # frozen for v1
    return model


def embed_images(df, image_encoder, batch_size=32):
    embeddings = np.zeros((len(df), IMAGE_EMBED_DIM), dtype=np.float32)
    valid_mask = np.zeros(len(df), dtype=bool)

    batch_imgs, batch_idxs = [], []

    def flush():
        if not batch_imgs:
            return
        with torch.no_grad():
            batch_tensor = torch.stack(batch_imgs)
            out = image_encoder(batch_tensor).numpy()  # DINOv2 returns (batch, 384) directly
        for i, idx in enumerate(batch_idxs):
            embeddings[idx] = out[i]
            valid_mask[idx] = True
        batch_imgs.clear()
        batch_idxs.clear()

    for i, path in enumerate(df["thumbnail_path"]):
        if not isinstance(path, str) or not os.path.exists(path):
            continue  # missing/failed thumbnail download -- row excluded downstream
        try:
            img = Image.open(path).convert("RGB")
            batch_imgs.append(IMAGE_TRANSFORM(img))
            batch_idxs.append(i)
        except Exception as e:
            print(f"  [warn] failed to load {path}: {e}")
            continue
        if len(batch_imgs) >= batch_size:
            flush()
    flush()

    return embeddings, valid_mask


def embed_titles(df, text_encoder, batch_size=64):
    titles = df["title"].fillna("").tolist()
    embeddings = text_encoder.encode(titles, batch_size=batch_size, show_progress_bar=True)
    return embeddings.astype(np.float32)


def main():
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    print(f"Loading {len(df)} rows from {CSV_PATH}")

    print("Building image encoder (frozen ConvNeXt-tiny)...")
    image_encoder = build_image_encoder()

    print("Embedding thumbnails...")
    image_embeddings, valid_image_mask = embed_images(df, image_encoder)

    print("Loading title encoder (MiniLM)...")
    text_encoder = SentenceTransformer(TEXT_MODEL_NAME)

    print("Embedding titles...")
    text_embeddings = embed_titles(df, text_encoder)

    np.save(os.path.join(EMBEDDINGS_DIR, "image_embeddings.npy"), image_embeddings)
    np.save(os.path.join(EMBEDDINGS_DIR, "text_embeddings.npy"), text_embeddings)
    np.save(os.path.join(EMBEDDINGS_DIR, "valid_image_mask.npy"), valid_image_mask)
    df[["video_id"]].to_csv(os.path.join(EMBEDDINGS_DIR, "video_id_order.csv"), index=False)

    print(f"\nDone. {valid_image_mask.sum()}/{len(df)} rows have valid image embeddings.")
    print(f"Saved to {EMBEDDINGS_DIR}/")


if __name__ == "__main__":
    main()