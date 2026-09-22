"""
Trains the late fusion head on cached embeddings + tabular features.

Usage:
    python -m models.precompute_embeddings   # run once, or after adding new data
    python -m models.train

NOTE: not executed end-to-end in the environment that generated this file
(no disk space to install torch here). The model architecture's tensor
shapes were checked by hand and via the standalone shape test in
late_fusion_model.py -- run that first if anything errors here.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from features.target import compute_target, invert_target
from models.dataset import VideoDataset, build_tabular_matrix, TABULAR_LOG_COLS, TABULAR_NUMERIC_COLS, TABULAR_BOOL_COLS
from models.late_fusion_model import LateFusionModel

CSV_PATH = "data/videos.csv"
VISUAL_FEATURES_PATH = "data/visual_features.csv"
EMBEDDINGS_DIR = "data/embeddings"
CHECKPOINT_PATH = "models/checkpoints/late_fusion_v1.pt"

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
EARLY_STOP_PATIENCE = 5  # stop if val_loss hasn't improved in this many epochs


def load_data():
    df = pd.read_csv(CSV_PATH)
    df["label_finalized"] = df["label_finalized"].astype(str) == "True"

    if not os.path.exists(VISUAL_FEATURES_PATH):
        raise RuntimeError(
            f"{VISUAL_FEATURES_PATH} not found -- run "
            "`python -m models.precompute_visual_features` first."
        )
    visual = pd.read_csv(VISUAL_FEATURES_PATH)
    df = df.merge(visual, on="video_id", how="left")  # left join -- rows not yet processed become NaN, filtered below

    image_embeddings = np.load(os.path.join(EMBEDDINGS_DIR, "image_embeddings.npy"))
    text_embeddings = np.load(os.path.join(EMBEDDINGS_DIR, "text_embeddings.npy"))
    valid_image_mask = np.load(os.path.join(EMBEDDINGS_DIR, "valid_image_mask.npy"))
    video_id_order = pd.read_csv(os.path.join(EMBEDDINGS_DIR, "video_id_order.csv"))

    df = df.merge(video_id_order.reset_index().rename(columns={"index": "_embed_idx"}), on="video_id")
    df = df.sort_values("_embed_idx").reset_index(drop=True)

    # Only rows that are: finalized, have a valid image embedding, have a real
    # trailing-views value, and have visual features already computed
    mask = (
        df["label_finalized"]
        & valid_image_mask[df["_embed_idx"].values]
        & df["trailing_avg_views"].notna()
        & df["has_face"].notna()
    )
    print(f"Using {mask.sum()}/{len(df)} rows after filtering "
          f"(finalized + valid image + has trailing_avg_views + has visual features)")

    df = df[mask].reset_index(drop=True)
    idxs = df["_embed_idx"].values
    image_embeddings = image_embeddings[idxs]
    text_embeddings = text_embeddings[idxs]

    # Safe to convert now -- every remaining row already passed the notna()
    # check above, so no NaN-to-string ambiguity to worry about here
    df["has_face"] = df["has_face"].astype(str) == "True"
    df["has_text_overlay"] = df["has_text_overlay"].astype(str) == "True"

    df["target"] = compute_target(df["views"], df["trailing_avg_views"])

    return df, image_embeddings, text_embeddings


def time_based_split(df):
    df = df.sort_values("published_at").reset_index(drop=True)
    n = len(df)
    train_end = int(n * (1 - VAL_FRACTION - TEST_FRACTION))
    val_end = int(n * (1 - TEST_FRACTION))
    return (
        df.iloc[:train_end].index.values,
        df.iloc[train_end:val_end].index.values,
        df.iloc[val_end:].index.values,
    )


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        pred = model(
            batch["image_embedding"].to(device),
            batch["text_embedding"].to(device),
            batch["tabular"].to(device),
        )
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
        pred = model(
            batch["image_embedding"].to(device),
            batch["text_embedding"].to(device),
            batch["tabular"].to(device),
        )
        loss = loss_fn(pred, batch["target"].to(device))
        total_loss += loss.item() * len(batch["target"])
        all_preds.extend(pred.cpu().numpy().tolist())
        all_targets.extend(batch["target"].numpy().tolist())
    return total_loss / len(loader.dataset), np.array(all_preds), np.array(all_targets)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, image_embeddings, text_embeddings = load_data()

    train_idx, val_idx, test_idx = time_based_split(df)
    print(f"Split (time-based): train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    genre_categories = sorted(df.iloc[train_idx]["genre"].dropna().unique().tolist())

    train_tabular, scaler = build_tabular_matrix(df.iloc[train_idx], genre_categories, fit_scaler=True)
    val_tabular, _ = build_tabular_matrix(df.iloc[val_idx], genre_categories, scaler=scaler)
    test_tabular, _ = build_tabular_matrix(df.iloc[test_idx], genre_categories, scaler=scaler)

    def make_dataset(idx, tabular):
        sub = df.iloc[idx]
        return VideoDataset(
            image_embeddings[idx], text_embeddings[idx], tabular,
            sub["target"].values, sub["video_id"].values,
        )

    train_ds = make_dataset(train_idx, train_tabular)
    val_ds = make_dataset(val_idx, val_tabular)
    test_ds = make_dataset(test_idx, test_tabular)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = LateFusionModel(
        image_dim=image_embeddings.shape[1],
        text_dim=text_embeddings.shape[1],
        tabular_dim=train_tabular.shape[1],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    loss_fn = torch.nn.HuberLoss()  # more robust to view-count outliers than MSE

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, _, _ = evaluate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "genre_categories": genre_categories,
                "scaler": scaler,
                "image_dim": image_embeddings.shape[1],
                "text_dim": text_embeddings.shape[1],
                "tabular_dim": train_tabular.shape[1],
            }, CHECKPOINT_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"No val_loss improvement in {EARLY_STOP_PATIENCE} epochs -- stopping early.")
                break

    # Final test evaluation using the best checkpoint, not just the last epoch
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, preds, targets = evaluate(model, test_loader, loss_fn, device)

    spearman_corr, _ = spearmanr(preds, targets)

    test_sub = df.iloc[test_idx]
    predicted_views = invert_target(preds, test_sub["trailing_avg_views"].values)
    actual_views = test_sub["views"].values
    rmse_views = np.sqrt(np.mean((predicted_views - actual_views) ** 2))

    print(f"\n--- Test results ---")
    print(f"Test loss (Huber, on target scale): {test_loss:.4f}")
    print(f"Spearman correlation (predicted vs actual relative performance): {spearman_corr:.4f}")
    print(f"RMSE in original view-count space: {rmse_views:,.0f}")


if __name__ == "__main__":
    main()