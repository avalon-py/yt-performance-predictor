"""
Precompute frozen image and title embeddings ONCE, cache to disk. Encoders are
frozen, so re-running them every epoch would be pure waste.

Usage:
    python -m models.precompute_embeddings                                              # CLIP ViT-B/32 (default)
    python -m models.precompute_embeddings --image-encoder dinov2                       # DINOv2
    python -m models.precompute_embeddings --image-encoder clip_b32 --image-mode crop
    python -m models.precompute_embeddings --image-encoder clip_b32 --limit 200         # Smoke test for 200 rows

Each run writes to its own folder, data/embeddings/<encoder>[_crop][_smoketest]/,
plus a meta.json, so encoders never overwrite each other.

NOTE: not executed here (no access to pretrained weights). Run with --limit
first and check the printed shapes and scales before a full run.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision import transforms
from torchvision.transforms import InterpolationMode

CSV_PATH = "data/videos.csv"
EMBEDDINGS_ROOT = "data/embeddings"
TEXT_MODEL_NAME = "all-MiniLM-L6-v2"

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# rescale=True: L2-normalize, then multiply by sqrt(dim) so each dimension has
# RMS ~1. Unit-norm CLIP vectors have per-dim values around 0.04, which would
# be swamped by EMBEDDING_NOISE_STD=0.02 and by the very low learning rate.
# Cosine similarity is unaffected by this scaling.
ENCODERS = {
    "dinov2": dict(kind="dinov2", dim=384, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                   interp=InterpolationMode.BILINEAR, rescale=False),
    "clip_b32": dict(kind="clip", hf_name="openai/clip-vit-base-patch32", dim=512,
                     mean=CLIP_MEAN, std=CLIP_STD, interp=InterpolationMode.BICUBIC, rescale=True),
    "clip_b16": dict(kind="clip", hf_name="openai/clip-vit-base-patch16", dim=512,
                     mean=CLIP_MEAN, std=CLIP_STD, interp=InterpolationMode.BICUBIC, rescale=True),
}


def build_transform(cfg, mode):
    if mode == "squash":  # whole frame, 16:9 squeezed to square (your original behaviour)
        resize = [transforms.Resize((224, 224), interpolation=cfg["interp"])]
    else:  # "crop": short side to 224, then center crop (CLIP's default; loses the sides)
        resize = [transforms.Resize(224, interpolation=cfg["interp"]), transforms.CenterCrop(224)]
    return transforms.Compose(
        resize + [transforms.ToTensor(), transforms.Normalize(cfg["mean"], cfg["std"])]
    )


def build_image_encoder(cfg, device):
    """Returns (encode_fn, clip_model_or_None). Encoders are frozen."""
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


def embed_images(df, encode, transform, cfg, device, batch_size=32):
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

    for i, path in enumerate(df["thumbnail_path"]):
        if not isinstance(path, str) or not os.path.exists(path):
            continue
        try:
            img = Image.open(path).convert("RGB")
            batch_imgs.append(transform(img))
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


def embed_titles_clip(df, clip_model, hf_name, device, batch_size=256):
    """Titles through CLIP's own text tower, same space as the CLIP image embeddings."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-encoder", choices=list(ENCODERS), default="clip_b32")
    parser.add_argument("--image-mode", choices=["squash", "crop"], default="squash")
    parser.add_argument("--limit", type=int, default=None, help="only embed the first N rows (smoke test)")
    args = parser.parse_args()

    cfg = ENCODERS[args.image_encoder]
    out_name = (args.image_encoder
                + ("_crop" if args.image_mode == "crop" else "")
                + ("_smoketest" if args.limit else ""))
    out_dir = os.path.join(EMBEDDINGS_ROOT, out_name)
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(CSV_PATH)
    if args.limit:
        df = df.head(args.limit)
    print(f"Loaded {len(df)} rows from {CSV_PATH}; device={device}; output -> {out_dir}")

    print(f"Building image encoder ({args.image_encoder}, frozen)...")
    encode, clip_model = build_image_encoder(cfg, device)
    transform = build_transform(cfg, args.image_mode)

    print("Embedding thumbnails...")
    image_embeddings, valid_mask = embed_images(df, encode, transform, cfg, device)
    image_embeddings = postprocess(image_embeddings, cfg)

    print("Loading title encoder (MiniLM)...")
    text_encoder = SentenceTransformer(TEXT_MODEL_NAME)
    print("Embedding titles...")
    text_embeddings = embed_titles(df, text_encoder)

    np.save(os.path.join(out_dir, "image_embeddings.npy"), image_embeddings)
    np.save(os.path.join(out_dir, "text_embeddings.npy"), text_embeddings)
    np.save(os.path.join(out_dir, "valid_image_mask.npy"), valid_mask)
    df[["video_id"]].to_csv(os.path.join(out_dir, "video_id_order.csv"), index=False)

    if clip_model is not None:
        print("Embedding titles with CLIP's text tower (for thumbnail-title similarity)...")
        clip_text = postprocess(embed_titles_clip(df, clip_model, cfg["hf_name"], device), cfg)
        np.save(os.path.join(out_dir, "clip_text_embeddings.npy"), clip_text)

    meta = {
        "image_encoder": args.image_encoder,
        "image_mode": args.image_mode,
        "image_dim": int(image_embeddings.shape[1]),
        "rescaled_to_unit_rms": cfg["rescale"],
        "text_encoder": TEXT_MODEL_NAME,
        "text_dim": int(text_embeddings.shape[1]),
        "n_rows": int(len(df)),
        "n_valid_images": int(valid_mask.sum()),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. {valid_mask.sum()}/{len(df)} rows have valid image embeddings.")
    print(f"image embeddings: shape={image_embeddings.shape}, "
          f"mean per-dim std={image_embeddings[valid_mask].std(axis=0).mean():.3f}")
    print(f"text embeddings:  shape={text_embeddings.shape}, "
          f"mean per-dim std={text_embeddings.std(axis=0).mean():.3f}")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()