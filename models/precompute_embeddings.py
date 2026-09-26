"""
Precompute CLIP image + title embeddings for videos missing them, writing
directly into videos.image_embedding / videos.text_embedding in Postgres.
Thumbnails are read from MinIO. Idempotent -- only processes rows where an
embedding is still NULL.

NOTE: ENCODERS / build_transform / build_image_encoder / postprocess /
embed_titles_clip / load_clip_text_tower are UNCHANGED from the original
dual-variant file -- serving/bundle.py and serving/features.py import these
by name as their single source of truth for encoder preprocessing. Only
main() and embed_images() (Postgres/MinIO orchestration, used only by this
script) have changed.

Usage:
    python -m models.precompute_embeddings
    python -m models.precompute_embeddings --limit 200   # smoke test
"""

import argparse
import io
import os

import boto3
import numpy as np
import pandas as pd
import torch
from botocore.config import Config
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# --- unchanged from the original file: serving/bundle.py and
# serving/features.py import these by name. ---
ENCODERS = {
    "dinov2": dict(kind="dinov2", dim=384, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                   interp=InterpolationMode.BILINEAR, rescale=False),
    "clip_b32": dict(kind="clip", hf_name="openai/clip-vit-base-patch32", dim=512,
                     mean=CLIP_MEAN, std=CLIP_STD, interp=InterpolationMode.BICUBIC, rescale=True),
    "clip_b16": dict(kind="clip", hf_name="openai/clip-vit-base-patch16", dim=512,
                     mean=CLIP_MEAN, std=CLIP_STD, interp=InterpolationMode.BICUBIC, rescale=True),
}


def build_transform(cfg, mode):
    if mode == "squash":
        resize = [transforms.Resize((224, 224), interpolation=cfg["interp"])]
    else:
        resize = [transforms.Resize(224, interpolation=cfg["interp"]), transforms.CenterCrop(224)]
    return transforms.Compose(
        resize + [transforms.ToTensor(), transforms.Normalize(cfg["mean"], cfg["std"])]
    )


def build_image_encoder(cfg, device):
    if cfg["kind"] == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        clip_model = None

        def encode(x):
            return model(x)
    else:
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained(cfg["hf_name"])
        clip_model = model

        def encode(x):
            pooled = model.vision_model(pixel_values=x).pooler_output
            return model.visual_projection(pooled)

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return encode, clip_model


def postprocess(emb, cfg):
    if cfg["rescale"]:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-8) * np.sqrt(emb.shape[1])
    return emb.astype(np.float32)


def load_clip_text_tower(hf_name, device):
    from transformers import CLIPModel
    model = CLIPModel.from_pretrained(hf_name)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


def embed_titles_clip(df, clip_model, hf_name, device, batch_size=256):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    titles = df["title"].fillna("").tolist()
    chunks = []
    for i in range(0, len(titles), batch_size):
        enc = tokenizer(titles[i:i + batch_size], padding=True, truncation=True,
                        max_length=77, return_tensors="pt").to(device)
        with torch.inference_mode():
            pooled = clip_model.text_model(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
            ).pooler_output
            chunks.append(clip_model.text_projection(pooled).float().cpu().numpy())
    return np.concatenate(chunks)
# --- end unchanged section ---


DB_URL = (
    f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:"
    f"{os.environ['POSTGRES_PASSWORD']}@localhost:5432/{os.environ['POSTGRES_DB']}"
)
engine = create_engine(DB_URL)

MINIO_BUCKET = "thumbnails"
minio_client = boto3.client(
    "s3",
    endpoint_url=f"http://{os.environ['MINIO_ENDPOINT']}",
    aws_access_key_id=os.environ["MINIO_ROOT_USER"],
    aws_secret_access_key=os.environ["MINIO_ROOT_PASSWORD"],
    config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
)

IMAGE_ENCODER = "clip_b32"  # only this script's own default choice -- ENCODERS
                            # dict itself still holds all three, unchanged


def to_vector_literal(values):
    if values is None:
        return None
    return "[" + ",".join(f"{v:.6f}" for v in values) + "]"


def fetch_thumbnail_image(object_key):
    if not isinstance(object_key, str):
        return None
    try:
        obj = minio_client.get_object(Bucket=MINIO_BUCKET, Key=object_key)
        return Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")
    except Exception as e:
        print(f"  [warn] failed to load {object_key}: {e}")
        return None


def embed_images(df, encode, transform, cfg, device, batch_size=32):
    """Same shape as the original, source of images changed from local disk
    to MinIO. Not imported by serving -- serving/features.py's encode_image()
    mirrors this logic for a single image but calls build_transform/postprocess
    directly, so this function changing internally doesn't affect serving."""
    embeddings = np.zeros((len(df), cfg["dim"]), dtype=np.float32)
    valid_mask = np.zeros(len(df), dtype=bool)
    batch_imgs, batch_idxs = [], []

    def flush():
        if not batch_imgs:
            return
        with torch.inference_mode():
            x = torch.stack(batch_imgs).to(device)
            out = encode(x).float().cpu().numpy()
        embeddings[batch_idxs] = out
        valid_mask[batch_idxs] = True
        batch_imgs.clear()
        batch_idxs.clear()

    for i, object_key in enumerate(df["thumbnail_path"]):
        img = fetch_thumbnail_image(object_key)
        if img is None:
            continue
        batch_imgs.append(transform(img))
        batch_idxs.append(i)
        if len(batch_imgs) >= batch_size:
            flush()
    flush()
    return embeddings, valid_mask


def update_embeddings(df, image_emb, valid_mask, text_emb):
    with engine.begin() as conn:
        for i, row in df.reset_index(drop=True).iterrows():
            conn.execute(
                text("""
                    UPDATE videos
                    SET image_embedding = :image_embedding,
                        text_embedding = :text_embedding
                    WHERE video_id = :video_id
                """),
                {
                    "video_id": row["video_id"],
                    "image_embedding": to_vector_literal(image_emb[i]) if valid_mask[i] else None,
                    "text_embedding": to_vector_literal(text_emb[i]),
                },
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="only embed the first N pending rows (smoke test)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ENCODERS[IMAGE_ENCODER]

    query = ("SELECT video_id, title, thumbnail_path FROM videos "
              "WHERE image_embedding IS NULL OR text_embedding IS NULL")
    if args.limit:
        query += f" LIMIT {args.limit}"
    df = pd.read_sql(query, engine)
    print(f"{len(df)} rows pending embeddings (device={device})")

    if df.empty:
        print("Nothing to do.")
        return

    print(f"Building image encoder ({IMAGE_ENCODER}, frozen)...")
    encode, clip_model = build_image_encoder(cfg, device)
    transform = build_transform(cfg, "squash")

    print("Embedding thumbnails (from MinIO)...")
    image_emb, valid_mask = embed_images(df, encode, transform, cfg, device)
    image_emb = postprocess(image_emb, cfg)

    print("Embedding titles (CLIP text tower)...")
    text_emb = postprocess(embed_titles_clip(df, clip_model, cfg["hf_name"], device), {"rescale": True})

    print("Writing embeddings back to Postgres...")
    update_embeddings(df, image_emb, valid_mask, text_emb)

    print(f"\nDone. {valid_mask.sum()}/{len(df)} rows got a valid image embedding "
          f"({len(df) - valid_mask.sum()} thumbnails failed to load).")


if __name__ == "__main__":
    main()