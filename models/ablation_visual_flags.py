"""
Targeted ablation: do the face / text-overlay tabular features earn their weight?

models/ablation_modalities.py bundles the visual tabular features
(face_count, has_face, has_text_overlay, color stats) together with the image
embedding, so it can't say what MTCNN and EasyOCR contribute on their own.
This script keeps ALL embeddings and every other feature fixed, and only
removes columns from the tabular branch:

  full               -- everything (identical to train.py)
  no_face_ocr        -- drop face_count, has_face, has_text_overlay
                        (keeps the cheap numpy color stats)
  no_visual_tabular  -- also drop the four color-stat columns

Each variant is trained on the same rows and the same chronological split with
the same seeds, so the per-seed differences are paired. Single-seed differences
of ~0.02 Spearman are within noise for this dataset, hence the multiple seeds.

Usage (same env vars as train.py: IMAGE_ENCODER, TEXT_ENCODER, USE_SIM):
    python -m models.ablation_visual_flags            # 5 seeds
    python -m models.ablation_visual_flags --seeds 10
"""

import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import models.dataset as ds
from models.dataset import VideoDataset, build_tabular_matrix
from models.late_fusion_model import LateFusionModel
from models.train import (
    load_data, time_based_split, set_seed, train_epoch, evaluate,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, EARLY_STOP_PATIENCE,
    WEIGHT_DECAY, DROPOUT, EMBEDDING_NOISE_STD,
)

FACE_OCR = ["face_count", "has_face", "has_text_overlay"]
COLOR = ["mean_saturation", "mean_brightness", "brightness_std", "warm_hue_ratio"]

VARIANTS = {
    "full": [],
    "no_face_ocr": FACE_OCR,
    "no_visual_tabular": FACE_OCR + COLOR,
}

RESULTS_PATH = "experiments/ablation_visual_flags.jsonl"


def set_columns(base_numeric, base_bool, drop):
    """Edit the shared column lists in place: build_tabular_matrix reads these
    module-level lists, and train.py imported the very same list objects."""
    ds.TABULAR_NUMERIC_COLS[:] = [c for c in base_numeric if c not in drop]
    ds.TABULAR_BOOL_COLS[:] = [c for c in base_bool if c not in drop]


def run_once(df, image_emb, text_emb, splits, seed, device):
    train_idx, val_idx, test_idx = splits
    set_seed(seed)

    genres = sorted(df.iloc[train_idx]["genre"].dropna().unique().tolist())
    tr_tab, scaler = build_tabular_matrix(df.iloc[train_idx], genres, fit_scaler=True)
    va_tab, _ = build_tabular_matrix(df.iloc[val_idx], genres, scaler=scaler)
    te_tab, _ = build_tabular_matrix(df.iloc[test_idx], genres, scaler=scaler)

    def make_loader(idx, tab, shuffle=False):
        sub = df.iloc[idx]
        dataset = VideoDataset(image_emb[idx], text_emb[idx], tab,
                               sub["target"].values, sub["video_id"].values)
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(train_idx, tr_tab, shuffle=True)
    val_loader = make_loader(val_idx, va_tab)
    test_loader = make_loader(test_idx, te_tab)

    model = LateFusionModel(
        image_dim=image_emb.shape[1], text_dim=text_emb.shape[1],
        tabular_dim=tr_tab.shape[1],
        dropout=DROPOUT, embedding_noise_std=EMBEDDING_NOISE_STD,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.HuberLoss()

    best_val, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, _, _ = evaluate(model, val_loader, loss_fn, device)
        if val_loss < best_val:
            best_val, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= EARLY_STOP_PATIENCE:
                break

    model.load_state_dict(best_state)
    _, preds, targets = evaluate(model, test_loader, loss_fn, device)
    spearman, _ = spearmanr(preds, targets)
    labels = (targets > 0).astype(int)
    auc = roc_auc_score(labels, preds) if len(np.unique(labels)) == 2 else float("nan")
    return {"spearman": float(spearman), "auc": float(auc),
            "best_epoch": best_epoch, "tabular_dim": int(tr_tab.shape[1])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, image_emb, text_emb = load_data()   # may append clip_sim to the column list
    splits = time_based_split(df)

    base_numeric = list(ds.TABULAR_NUMERIC_COLS)
    base_bool = list(ds.TABULAR_BOOL_COLS)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    rows = []
    for seed in range(1, args.seeds + 1):
        for name, drop in VARIANTS.items():
            set_columns(base_numeric, base_bool, drop)
            res = run_once(df, image_emb, text_emb, splits, seed, device)
            row = {"variant": name, "seed": seed, "n_test": len(splits[2]), **res}
            rows.append(row)
            print(f"seed={seed} {name:18s} spearman={res['spearman']:.4f} "
                  f"auc={res['auc']:.4f} dim={res['tabular_dim']} epoch={res['best_epoch']}")
            with open(RESULTS_PATH, "a") as f:
                f.write(json.dumps(row) + "\n")
    set_columns(base_numeric, base_bool, [])  # restore

    res_df = pd.DataFrame(rows)
    summary = res_df.groupby("variant")[["spearman", "auc"]].agg(["mean", "std"])
    print("\n=== Mean / std over seeds (same rows, same split) ===")
    print(summary.loc[list(VARIANTS)].to_string(float_format=lambda x: f"{x:.4f}"))

    wide = res_df.pivot(index="seed", columns="variant", values="spearman")
    print("\n=== Paired difference in Spearman vs. full (variant - full), over seeds ===")
    for name in VARIANTS:
        if name == "full":
            continue
        d = wide[name] - wide["full"]
        print(f"{name:18s} mean={d.mean():+.4f}  std={d.std():.4f}  "
              f"variant>=full in {(d >= 0).sum()}/{len(d)} seeds")


if __name__ == "__main__":
    main()