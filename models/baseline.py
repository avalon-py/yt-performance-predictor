"""
Compares three models on the identical time-based test split, all metrics
computed the same way for a fair side-by-side read:
  1. Naive       -- always predict target=0 ("video matches channel average")
  2. Linear       -- target ~ log1p(trailing_avg_views), simple linear regression
  3. Fusion model -- your trained late-fusion net, loaded from its checkpoint

This answers the question a lone Spearman number can't: how much of the
fusion model's apparent skill is actually coming from the image/text
branches, versus something this simple could already capture from
trailing_avg_views alone.

Usage:
    python -m models.baseline
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, roc_auc_score
from scipy.stats import spearmanr

from features.target import invert_target
from models.train import load_data, time_based_split, CHECKPOINT_PATH
from models.dataset import VideoDataset, build_tabular_matrix
from models.late_fusion_model import LateFusionModel


def compute_metrics(preds, targets, trailing_avg_views, actual_views):
    """One consistent metric set for any model's predictions on the target scale."""
    predicted_views = invert_target(preds, trailing_avg_views)

    rmse_views = np.sqrt(np.mean((predicted_views - actual_views) ** 2))
    mae_views = mean_absolute_error(actual_views, predicted_views)
    mape_views = mean_absolute_percentage_error(actual_views, predicted_views)

    # Spearman and AUC are undefined for a constant prediction (zero variance) --
    # report N/A instead of a misleading 0.0 or a default 0.5.
    if np.std(preds) < 1e-12:
        spearman_corr = float("nan")
        auc = float("nan")
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


def run_naive(test_df, y_test, actual_views):
    preds = np.zeros_like(y_test)
    return compute_metrics(preds, y_test, test_df["trailing_avg_views"].values, actual_views)


def run_linear(train_df, test_df, y_train, y_test, actual_views):
    X_train = np.log1p(train_df["trailing_avg_views"].values).reshape(-1, 1)
    X_test = np.log1p(test_df["trailing_avg_views"].values).reshape(-1, 1)

    model = LinearRegression().fit(X_train, y_train)
    preds = model.predict(X_test)
    return compute_metrics(preds, y_test, test_df["trailing_avg_views"].values, actual_views), model


def run_fusion_model(df, test_idx, image_embeddings, text_embeddings, y_test, actual_views, device):
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)

    test_df = df.iloc[test_idx]
    test_tabular, _ = build_tabular_matrix(
        test_df, checkpoint["genre_categories"], scaler=checkpoint["scaler"]
    )
    test_ds = VideoDataset(
        image_embeddings[test_idx], text_embeddings[test_idx], test_tabular,
        test_df["target"].values, test_df["video_id"].values,
    )
    test_loader = DataLoader(test_ds, batch_size=64)

    model = LateFusionModel(
        image_dim=checkpoint["image_dim"],
        text_dim=checkpoint["text_dim"],
        tabular_dim=checkpoint["tabular_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            pred = model(
                batch["image_embedding"].to(device),
                batch["text_embedding"].to(device),
                batch["tabular"].to(device),
            )
            all_preds.extend(pred.cpu().numpy().tolist())
    preds = np.array(all_preds)

    return compute_metrics(preds, y_test, test_df["trailing_avg_views"].values, actual_views), checkpoint


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, image_embeddings, text_embeddings = load_data()
    train_idx, val_idx, test_idx = time_based_split(df)

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    y_train = train_df["target"].values
    y_test = test_df["target"].values
    actual_views = test_df["views"].values

    results = {}
    results["Naive"] = run_naive(test_df, y_test, actual_views)
    results["Linear (trailing_avg_views)"], linear_model = run_linear(train_df, test_df, y_train, y_test, actual_views)
    results["Fusion Model"], fusion_checkpoint = run_fusion_model(
        df, test_idx, image_embeddings, text_embeddings, y_test, actual_views, device
    )

    table = pd.DataFrame(results)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

    print("\n=== Model comparison (identical test split) ===\n")
    print(table.to_string())

    print(f"\nLinear baseline coefficient: {linear_model.coef_[0]:.4f}, intercept: {linear_model.intercept_:.4f}")
    print(f"Fusion model checkpoint: epoch {fusion_checkpoint.get('epoch', '?')}, "
          f"val_loss={fusion_checkpoint.get('val_loss', float('nan')):.4f}")

    print("\nHow to read this:")
    print("- If Fusion Model isn't clearly ahead of Linear on Spearman/AUC, the image/text")
    print("  branches aren't earning their complexity over trailing_avg_views alone.")
    print("- RMSE/MAE/MAPE are in raw view-count space -- expect these to be noisy and")
    print("  outlier-sensitive given the channel-scale spread in this dataset.")


if __name__ == "__main__":
    main()