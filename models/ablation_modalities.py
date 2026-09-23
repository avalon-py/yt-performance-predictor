"""
Ablation: train the late-fusion architecture with different modalities
active, where "modality" means BOTH the engineered tabular features AND the
embedding for that source -- e.g. tabular_text includes title-derived
tabular columns (title_length_chars, title_has_question_mark, etc.) AND the
MiniLM text_embedding together, never one without the other.

Variants:
  tabular_only  -- no title-derived or video-derived features/embeddings at all
  tabular_text  -- title-derived tabular features + text_embedding; no video signal
  tabular_image -- video-derived tabular features + image_embedding; no title signal
  full_fusion   -- everything

Usage:
    python -m models.ablation_modalities
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, roc_auc_score

from models.train import (
    load_data, time_based_split,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, EARLY_STOP_PATIENCE,
    WEIGHT_DECAY, DROPOUT, EMBEDDING_NOISE_STD,
)
from features.target import invert_target

CHECKPOINT_DIR = "models/checkpoints/ablations"

TABULAR_LOG_COLS = ["subscriber_count_at_upload", "trailing_avg_views"]  # neutral -- always included
TABULAR_NEUTRAL_NUMERIC_COLS = ["duration_seconds"]

TITLE_NUMERIC_COLS = [
    "title_length_chars", "title_word_count", "title_capitalized_word_count",
    "title_capitalized_letter_count", "title_capitalized_letter_ratio", "title_symbol_count",
]
TITLE_BOOL_COLS = ["title_has_question_mark", "title_has_number"]

VIDEO_NUMERIC_COLS = ["face_count", "mean_saturation", "mean_brightness", "brightness_std", "warm_hue_ratio"]
VIDEO_BOOL_COLS = ["has_face", "has_text_overlay"]

VARIANTS = {
    "tabular_only":  {"include_title": False, "include_video": False, "use_image": False, "use_text": False},
    "tabular_text":  {"include_title": True,  "include_video": False, "use_image": False, "use_text": True},
    "tabular_image": {"include_title": False, "include_video": True,  "use_image": True,  "use_text": False},
    "full_fusion":   {"include_title": True,  "include_video": True,  "use_image": True,  "use_text": True},
}


def build_variant_tabular_matrix(df, genre_categories, include_title, include_video, scaler=None, fit_scaler=False):
    numeric_cols = list(TABULAR_NEUTRAL_NUMERIC_COLS)
    bool_cols = []
    if include_title:
        numeric_cols += TITLE_NUMERIC_COLS
        bool_cols += TITLE_BOOL_COLS
    if include_video:
        numeric_cols += VIDEO_NUMERIC_COLS
        bool_cols += VIDEO_BOOL_COLS

    log_vals = df[TABULAR_LOG_COLS].astype(float).apply(np.log1p).values
    numeric_vals = df[numeric_cols].astype(float).values

    if fit_scaler:
        log_scaler = StandardScaler().fit(log_vals)
        numeric_scaler = RobustScaler().fit(numeric_vals)
        scaler = (log_scaler, numeric_scaler)
    log_scaler, numeric_scaler = scaler

    log_scaled = log_scaler.transform(log_vals)
    numeric_scaled = numeric_scaler.transform(numeric_vals)

    bool_vals = df[bool_cols].astype(float).values if bool_cols else np.zeros((len(df), 0))

    genre_onehot = np.zeros((len(df), len(genre_categories)), dtype=np.float32)
    for i, genre in enumerate(df["genre"].values):
        if genre in genre_categories:
            genre_onehot[i, genre_categories.index(genre)] = 1.0

    tabular = np.concatenate([log_scaled, numeric_scaled, bool_vals, genre_onehot], axis=1).astype(np.float32)
    return tabular, scaler


class AblationDataset(Dataset):
    def __init__(self, tabular, targets, video_ids, image_embeddings=None, text_embeddings=None):
        self.tabular = torch.as_tensor(tabular, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.video_ids = video_ids
        self.image_embeddings = torch.as_tensor(image_embeddings, dtype=torch.float32) if image_embeddings is not None else None
        self.text_embeddings = torch.as_tensor(text_embeddings, dtype=torch.float32) if text_embeddings is not None else None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        item = {"tabular": self.tabular[idx], "target": self.targets[idx], "video_id": self.video_ids[idx]}
        if self.image_embeddings is not None:
            item["image_embedding"] = self.image_embeddings[idx]
        if self.text_embeddings is not None:
            item["text_embedding"] = self.text_embeddings[idx]
        return item


class AblationModel(nn.Module):
    def __init__(self, tabular_dim, use_image, use_text,
                 image_dim=None, text_dim=None,
                 proj_dim=64, tabular_proj_dim=32, dropout=0.2,
                 embedding_noise_std=0.02):
        super().__init__()
        self.use_image = use_image
        self.use_text = use_text
        self.embedding_noise_std = embedding_noise_std

        self.tabular_proj = nn.Sequential(
            nn.Linear(tabular_dim, tabular_proj_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        fusion_input_dim = tabular_proj_dim

        if use_image:
            self.image_proj = nn.Sequential(
                nn.Linear(image_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout),
            )
            fusion_input_dim += proj_dim
        if use_text:
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout),
            )
            fusion_input_dim += proj_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, tabular, image_embedding=None, text_embedding=None):
        parts = [self.tabular_proj(tabular)]
        if self.use_image:
            if self.training and self.embedding_noise_std > 0:
                image_embedding = image_embedding + torch.randn_like(image_embedding) * self.embedding_noise_std
            parts.append(self.image_proj(image_embedding))
        if self.use_text:
            if self.training and self.embedding_noise_std > 0:
                text_embedding = text_embedding + torch.randn_like(text_embedding) * self.embedding_noise_std
            parts.append(self.text_proj(text_embedding))
        fused = torch.cat(parts, dim=1)
        return self.fusion_head(fused).squeeze(-1)


def forward_batch(model, batch, device):
    kwargs = {"tabular": batch["tabular"].to(device)}
    if "image_embedding" in batch:
        kwargs["image_embedding"] = batch["image_embedding"].to(device)
    if "text_embedding" in batch:
        kwargs["text_embedding"] = batch["text_embedding"].to(device)
    return model(**kwargs)


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        pred = forward_batch(model, batch, device)
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
        pred = forward_batch(model, batch, device)
        loss = loss_fn(pred, batch["target"].to(device))
        total_loss += loss.item() * len(batch["target"])
        all_preds.extend(pred.cpu().numpy().tolist())
        all_targets.extend(batch["target"].numpy().tolist())
    return total_loss / len(loader.dataset), np.array(all_preds), np.array(all_targets)


def compute_metrics(preds, targets, trailing_avg_views, actual_views):
    predicted_views = invert_target(preds, trailing_avg_views)

    rmse_views = np.sqrt(np.mean((predicted_views - actual_views) ** 2))
    mae_views = mean_absolute_error(actual_views, predicted_views)
    mape_views = mean_absolute_percentage_error(actual_views, predicted_views)

    if np.std(preds) < 1e-12:
        spearman_corr, auc = float("nan"), float("nan")
    else:
        spearman_corr, _ = spearmanr(preds, targets)
        binary_labels = (targets > 0).astype(int)
        auc = roc_auc_score(binary_labels, preds) if len(np.unique(binary_labels)) == 2 else float("nan")

    return {
        "RMSE (views)": rmse_views,
        "MAE (views)": mae_views,
        "MAPE (views)": mape_views,
        "Spearman": spearman_corr,
        "AUC": auc,
    }


def run_variant(name, config, df, train_idx, val_idx, test_idx, image_embeddings, text_embeddings, device):
    genre_categories = sorted(df.iloc[train_idx]["genre"].dropna().unique().tolist())

    train_tabular, scaler = build_variant_tabular_matrix(
        df.iloc[train_idx], genre_categories, config["include_title"], config["include_video"], fit_scaler=True
    )
    val_tabular, _ = build_variant_tabular_matrix(
        df.iloc[val_idx], genre_categories, config["include_title"], config["include_video"], scaler=scaler
    )
    test_tabular, _ = build_variant_tabular_matrix(
        df.iloc[test_idx], genre_categories, config["include_title"], config["include_video"], scaler=scaler
    )

    use_image, use_text = config["use_image"], config["use_text"]

    def make_dataset(idx, tabular):
        sub = df.iloc[idx]
        return AblationDataset(
            tabular, sub["target"].values, sub["video_id"].values,
            image_embeddings=image_embeddings[idx] if use_image else None,
            text_embeddings=text_embeddings[idx] if use_text else None,
        )

    train_loader = DataLoader(make_dataset(train_idx, train_tabular), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(make_dataset(val_idx, val_tabular), batch_size=BATCH_SIZE)
    test_loader = DataLoader(make_dataset(test_idx, test_tabular), batch_size=BATCH_SIZE)

    model = AblationModel(
        tabular_dim=train_tabular.shape[1],
        use_image=use_image, use_text=use_text,
        image_dim=image_embeddings.shape[1] if use_image else None,
        text_dim=text_embeddings.shape[1] if use_text else None,
        dropout=DROPOUT, embedding_noise_std=EMBEDDING_NOISE_STD,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.HuberLoss()

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{name}.pt")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, _, _ = evaluate(model, val_loader, loss_fn, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                break

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, preds, targets = evaluate(model, test_loader, loss_fn, device)

    test_sub = df.iloc[test_idx]
    metrics = compute_metrics(preds, targets, test_sub["trailing_avg_views"].values, test_sub["views"].values)

    return {
        "variant": name,
        **metrics,
        # kept for debugging, not printed in the main table
        "_tabular_dim": train_tabular.shape[1],
        "_best_epoch": checkpoint["epoch"],
        "_best_val_loss": checkpoint["val_loss"],
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, image_embeddings, text_embeddings = load_data()
    train_idx, val_idx, test_idx = time_based_split(df)

    results = []
    for name, config in VARIANTS.items():
        print(f"\n=== Training variant: {name} ===")
        result = run_variant(name, config, df, train_idx, val_idx, test_idx, image_embeddings, text_embeddings, device)
        print(f"  Spearman={result['Spearman']:.4f}  AUC={result['AUC']:.4f}  "
              f"RMSE={result['RMSE (views)']:,.0f}  MAE={result['MAE (views)']:,.0f}  "
              f"MAPE={result['MAPE (views)']:.2%}")
        results.append(result)

    table = pd.DataFrame(results).set_index("variant")
    table = table[["RMSE (views)", "MAE (views)", "MAPE (views)", "Spearman", "AUC"]]
    table = table.sort_values("Spearman", ascending=False)

    print("\n=== Modality ablation: all metrics, identical test split ===\n")
    print(table.to_string(float_format=lambda x: f"{x:,.4f}"))


if __name__ == "__main__":
    main()