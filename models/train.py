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
from collections import deque
import matplotlib.pyplot as plt
import json
import random


from features.target import compute_target, invert_target
from models.dataset import VideoDataset, build_tabular_matrix, TABULAR_LOG_COLS, TABULAR_NUMERIC_COLS, TABULAR_BOOL_COLS
from models.late_fusion_model import LateFusionModel
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, roc_auc_score

IMAGE_ENCODER = os.environ.get("IMAGE_ENCODER", "dinov2")
SEED = int(os.environ.get("SEED", 42))
EMBEDDINGS_DIR = os.path.join("data/embeddings", IMAGE_ENCODER)
CHECKPOINT_PATH = f"models/checkpoints/late_fusion_v1_{IMAGE_ENCODER}.pt"
RESULTS_PATH = "experiments/results.jsonl"
CSV_PATH = "data/videos.csv"
VISUAL_FEATURES_PATH = "data/visual_features.csv"
PLOTS_DIR = "models/plots"

BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 2e-5
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
EARLY_STOP_PATIENCE = 10
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
EMBEDDING_NOISE_STD = 0.02
SPEARMAN_SMOOTHING_WINDOW = 5

def load_data():
    df = pd.read_csv(CSV_PATH)
    df["label_finalized"] = df["label_finalized"].astype(str) == "True"

    if not os.path.exists(VISUAL_FEATURES_PATH):
        raise RuntimeError(
            f"{VISUAL_FEATURES_PATH} not found -- run "
            "`python -m models.precompute_visual_features` first."
        )
    visual = pd.read_csv(VISUAL_FEATURES_PATH)
    df = df.merge(visual, on="video_id", how="left")

    image_embeddings = np.load(os.path.join(EMBEDDINGS_DIR, "image_embeddings.npy"))
    text_embeddings = np.load(os.path.join(EMBEDDINGS_DIR, "text_embeddings.npy"))
    valid_image_mask = np.load(os.path.join(EMBEDDINGS_DIR, "valid_image_mask.npy"))
    video_id_order = pd.read_csv(os.path.join(EMBEDDINGS_DIR, "video_id_order.csv"))

    df = df.merge(video_id_order.reset_index().rename(columns={"index": "_embed_idx"}), on="video_id")
    df = df.sort_values("_embed_idx").reset_index(drop=True)

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

    df["has_face"] = df["has_face"].astype(str) == "True"
    df["has_text_overlay"] = df["has_text_overlay"].astype(str) == "True"

    df["target"] = compute_target(df["views"], df["trailing_avg_views"])

    df.to_csv("data/full_dataset.csv")

    return df, image_embeddings, text_embeddings


def time_based_split(df):
    sorted_idx = df.sort_values("published_at").index
    n = len(sorted_idx)
    train_end = int(n * (1 - VAL_FRACTION - TEST_FRACTION))
    val_end = int(n * (1 - TEST_FRACTION))
    return (
        sorted_idx[:train_end].to_numpy(),
        sorted_idx[train_end:val_end].to_numpy(),
        sorted_idx[val_end:].to_numpy(),
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


def constant_mean_reference(train_targets, val_targets, loss_fn):
    """Huber loss of the 'predict the training mean for everything' baseline.
    If the trained model barely beats this on val loss, it's not learning
    much of a real relationship -- that's underfitting, not a data/architecture
    problem, and no amount of extra capacity or more data fixes it; the fix
    is loosening the optimization (higher LR, less regularization)."""
    const_pred = np.full_like(val_targets, fill_value=train_targets.mean(), dtype=np.float64)
    loss = loss_fn(torch.tensor(const_pred), torch.tensor(val_targets)).item()
    return loss


def plot_training_curves(train_losses, val_losses, val_spearmans, smoothed_spearmans,
                          best_epoch, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    epochs_range = range(1, len(train_losses) + 1)

    # --- Loss curve (overfitting indicator + actual stopping criterion) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_range, train_losses, label="train_loss")
    ax.plot(epochs_range, val_losses, label="val_loss")
    if best_epoch is not None:
        ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.6,
                   label=f"checkpoint (epoch {best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber loss")
    ax.set_title("Train vs Val Loss")
    ax.legend()
    fig.tight_layout()
    loss_path = os.path.join(plots_dir, "loss_curve.png")
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)

    # --- Spearman curve (reported for interpretation, not used for stopping) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_range, val_spearmans, label="val_spearman (raw)", alpha=0.4)
    smoothed_x = [e for e, s in zip(epochs_range, smoothed_spearmans) if s is not None]
    smoothed_y = [s for s in smoothed_spearmans if s is not None]
    if smoothed_y:
        ax.plot(smoothed_x, smoothed_y, label="val_spearman (smoothed)", linewidth=2)
    if best_epoch is not None:
        ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.6,
                   label=f"checkpoint (epoch {best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Spearman correlation")
    ax.set_title("Validation Spearman Correlation (diagnostic only -- not the stopping criterion)")
    ax.legend()
    fig.tight_layout()
    spearman_path = os.path.join(plots_dir, "spearman_curve.png")
    fig.savefig(spearman_path, dpi=150)
    plt.close(fig)

    print(f"\nSaved training curves to {loss_path} and {spearman_path}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed(SEED)

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
        dropout=DROPOUT,
        embedding_noise_std=EMBEDDING_NOISE_STD,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.HuberLoss()

    const_val_loss = constant_mean_reference(
        df.iloc[train_idx]["target"].values, df.iloc[val_idx]["target"].values, loss_fn
    )
    print(f"Reference: constant (train-mean) val_loss={const_val_loss:.4f}\n")

    best_val_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    spearman_window = deque(maxlen=SPEARMAN_SMOOTHING_WINDOW)
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    train_loss_history = []
    val_loss_history = []
    val_spearman_history = []
    smoothed_spearman_history = []

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

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_spearman_history.append(val_spearman)
        smoothed_spearman_history.append(smoothed_spearman)

        smoothed_str = f"{smoothed_spearman:.4f}" if smoothed_spearman is not None else "N/A"
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_spearman={val_spearman:.4f} | smoothed={smoothed_str}")

        # Pure val_loss-based checkpointing and early stopping: same quantity
        # the optimizer is minimizing, stable/low-variance in our curves,
        # and on the normalized log-ratio scale rather than raw view-count
        # space -- so a single viral outlier can't dominate the decision the
        # way it would if RMSE-in-views were used instead.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "genre_categories": genre_categories,
                "scaler": scaler,
                "image_dim": image_embeddings.shape[1],
                "text_dim": text_embeddings.shape[1],
                "tabular_dim": train_tabular.shape[1],
                "epoch": epoch,
                "val_loss": val_loss,
                "val_spearman_raw": val_spearman,
                "val_spearman_smoothed": smoothed_spearman,
                "image_encoder": IMAGE_ENCODER,
            }, CHECKPOINT_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"No val_loss improvement in {EARLY_STOP_PATIENCE} epochs -- stopping early.")
                break

    plot_training_curves(
        train_loss_history, val_loss_history,
        val_spearman_history, smoothed_spearman_history,
        best_epoch, PLOTS_DIR,
    )

    # Final test evaluation using the best checkpoint, not just the last epoch
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, preds, targets = evaluate(model, test_loader, loss_fn, device)

    spearman_corr, _ = spearmanr(preds, targets)

    test_sub = df.iloc[test_idx]
    predicted_views = invert_target(preds, test_sub["trailing_avg_views"].values)
    actual_views = test_sub["views"].values
    rmse_views = np.sqrt(np.mean((predicted_views - actual_views) ** 2))
    mae_target_scale = mean_absolute_error(targets, preds)
    mae_views = mean_absolute_error(actual_views, predicted_views)
    mape_views = mean_absolute_percentage_error(actual_views, predicted_views)
    binary_labels = (targets > 0).astype(int)
    if len(np.unique(binary_labels)) < 2:
        auc = float("nan")
        print("  [warn] test set has only one class (all over- or all under-performing) -- AUC undefined")
    else:
        auc = roc_auc_score(binary_labels, preds)

    print(f"Checkpoint was saved at epoch {checkpoint.get('epoch', '?')} "
          f"(val_loss={checkpoint.get('val_loss', float('nan')):.4f}, "
          f"val_spearman_raw={checkpoint.get('val_spearman_raw', float('nan')):.4f})")

    print(f"\n--- Test results (all downstream/diagnostic -- not used to select the checkpoint) ---")
    print(f"Test loss (Huber, on target scale): {test_loss:.4f}")
    print(f"Spearman correlation (predicted vs actual relative performance): {spearman_corr:.4f}")
    print(f"RMSE in original view-count space: {rmse_views:,.0f}")
    print(f"MAE (target scale): {mae_target_scale:.4f}")
    print(f"MAE (view-count scale): {mae_views:,.0f}")
    print(f"MAPE (view-count scale): {mape_views:.2%}")
    print(f"AUC (overperform vs underperform baseline): {auc:.4f}")
    print(f"(Reference: constant train-mean predictor val_loss was {const_val_loss:.4f} -- "
          f"if best val_loss during training was close to that, revisit LR/regularization.)")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps({
            "image_encoder": IMAGE_ENCODER, "seed": SEED,
            "best_epoch": checkpoint.get("epoch"),
            "val_loss": float(checkpoint.get("val_loss", float("nan"))),
            "test_loss": float(test_loss),
            "test_spearman": float(spearman_corr),
            "test_auc": float(auc),
            "test_mae_target": float(mae_target_scale),
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        }) + "\n")

    os.makedirs("experiments/preds", exist_ok=True)
    np.savez(f"experiments/preds/{IMAGE_ENCODER}_s{SEED}.npz",
             video_id=test_sub["video_id"].values, pred=preds, target=targets)


if __name__ == "__main__":
    main()