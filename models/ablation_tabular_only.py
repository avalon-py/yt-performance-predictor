"""
Ablation: train ONLY the tabular branch of the late-fusion architecture
(tabular_proj -> a fusion_head-shaped regression head), with no image/text
embeddings at all. Same training loop, same hyperparameters, same data
split as models/train.py -- the only thing that changes is the model.

Purpose: isolate whether the net's underperformance vs. the LightGBM
tabular-only baseline (0.4448 Spearman) is a training-loop/optimization
problem or a fusion-architecture problem (embeddings diluting the tabular
signal in the concat).

  - If this gets close to ~0.44 Spearman -> training loop is fine; the
    problem is specifically the fusion (256+256+128 concat drowning out
    the small tabular block relative to the two much larger embedding
    blocks).
  - If this still lands near ~0.30 -> the problem is in the
    optimization/architecture itself (LR, head capacity, gradient flow
    through the 16->1 bottleneck), independent of fusion.

Usage:
    python -m models.ablation_tabular_only
"""

import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr

from models.train import (
    load_data, time_based_split, constant_mean_reference,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, EARLY_STOP_PATIENCE,
    WEIGHT_DECAY, DROPOUT, SPEARMAN_SMOOTHING_WINDOW,
)
from models.dataset import build_tabular_matrix

CHECKPOINT_PATH = "models/checkpoints/ablation_tabular_only.pt"


class TabularOnlyDataset(Dataset):
    """Same shape/contract as VideoDataset, minus the embedding fields."""

    def __init__(self, tabular, targets, video_ids):
        self.tabular = torch.as_tensor(tabular, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.video_ids = video_ids

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "tabular": self.tabular[idx],
            "target": self.targets[idx],
            "video_id": self.video_ids[idx],
        }


class TabularOnlyModel(nn.Module):
    """
    Mirrors LateFusionModel's tabular_proj -> fusion_head path exactly
    (same layer shapes/dropout), just without concatenating image/text
    projections in. This keeps the comparison to the full model apples-
    to-apples on everything except the presence of the embedding branches.
    """

    def __init__(self, tabular_dim, tabular_proj_dim=128, dropout=0.2):
        super().__init__()
        self.tabular_proj = nn.Sequential(
            nn.Linear(tabular_dim, tabular_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(tabular_proj_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, tabular):
        tab = self.tabular_proj(tabular)
        return self.fusion_head(tab).squeeze(-1)


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        pred = model(batch["tabular"].to(device))
        loss = loss_fn(pred, batch["target"].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch["target"])
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for batch in loader:
        pred = model(batch["tabular"].to(device))
        loss = loss_fn(pred, batch["target"].to(device))
        total_loss += loss.item() * len(batch["target"])
        all_preds.extend(pred.cpu().numpy().tolist())
        all_targets.extend(batch["target"].numpy().tolist())
    return total_loss / len(loader.dataset), np.array(all_preds), np.array(all_targets)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reuse load_data()/time_based_split() unchanged so the split and
    # filtering are identical to every other run -- image/text embeddings
    # come back too but are simply never used below.
    df, _image_embeddings, _text_embeddings = load_data()
    train_idx, val_idx, test_idx = time_based_split(df)
    print(f"Split (time-based): train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    genre_categories = sorted(df.iloc[train_idx]["genre"].dropna().unique().tolist())
    train_tabular, scaler = build_tabular_matrix(df.iloc[train_idx], genre_categories, fit_scaler=True)
    val_tabular, _ = build_tabular_matrix(df.iloc[val_idx], genre_categories, scaler=scaler)
    test_tabular, _ = build_tabular_matrix(df.iloc[test_idx], genre_categories, scaler=scaler)

    def make_dataset(idx, tabular):
        sub = df.iloc[idx]
        return TabularOnlyDataset(tabular, sub["target"].values, sub["video_id"].values)

    train_ds = make_dataset(train_idx, train_tabular)
    val_ds = make_dataset(val_idx, val_tabular)
    test_ds = make_dataset(test_idx, test_tabular)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = TabularOnlyModel(
        tabular_dim=train_tabular.shape[1],
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.HuberLoss()

    const_val_loss = constant_mean_reference(
        df.iloc[train_idx]["target"].values, df.iloc[val_idx]["target"].values, loss_fn
    )
    print(f"Reference: constant (train-mean) val_loss={const_val_loss:.4f}\n")
    print(f"Reference: LightGBM tabular-only test Spearman was 0.4448 (from inspect_target_and_baseline.py)\n")

    best_smoothed_spearman = -float("inf")
    epochs_without_improvement = 0
    spearman_window = deque(maxlen=SPEARMAN_SMOOTHING_WINDOW)
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_preds, val_targets = evaluate(model, val_loader, loss_fn, device)
        val_spearman, _ = spearmanr(val_preds, val_targets)

        spearman_window.append(val_spearman)
        smoothed_spearman = (
            sum(spearman_window) / len(spearman_window)
            if len(spearman_window) == SPEARMAN_SMOOTHING_WINDOW
            else None
        )
        smoothed_str = f"{smoothed_spearman:.4f}" if smoothed_spearman is not None else "N/A"
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_spearman={val_spearman:.4f} | smoothed={smoothed_str}")

        if smoothed_spearman is None:
            continue

        if smoothed_spearman > best_smoothed_spearman:
            best_smoothed_spearman = smoothed_spearman
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "genre_categories": genre_categories,
                "scaler": scaler,
                "tabular_dim": train_tabular.shape[1],
                "epoch": epoch,
                "val_spearman_smoothed": smoothed_spearman,
            }, CHECKPOINT_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"No smoothed val_spearman improvement in {EARLY_STOP_PATIENCE} epochs -- stopping early.")
                break

    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, preds, targets = evaluate(model, test_loader, loss_fn, device)
    test_spearman, _ = spearmanr(preds, targets)

    print(f"\n--- Ablation test results (tabular-only net) ---")
    print(f"Checkpoint from epoch {checkpoint['epoch']} (smoothed val_spearman={checkpoint['val_spearman_smoothed']:.4f})")
    print(f"Test loss (Huber): {test_loss:.4f}")
    print(f"Test Spearman: {test_spearman:.4f}")
    print(f"\nCompare against:")
    print(f"  LightGBM tabular-only:        0.4448")
    print(f"  Full late-fusion net:         0.3010")
    print(f"  This ablation (tabular-only net, same training loop): {test_spearman:.4f}")
    print(f"\nInterpretation:")
    print(f"  - close to 0.44 -> training loop is fine; fusion/concat with embeddings is the problem")
    print(f"  - still near 0.30 -> optimization/architecture problem, independent of fusion")


if __name__ == "__main__":
    main()