"""
Baseline: predict the SAME target (log-relative performance vs. channel
baseline) using ONLY trailing_avg_views as a feature -- no thumbnail, no
title, no other tabular features at all.

This answers the question your fusion model's Spearman number can't answer
on its own: how much of that correlation is coming from the image/text
branches, versus something this trivially simple could already capture?

Two reference points reported, both on the identical time-based split:
  1. Naive: always predict target=0 (i.e. "video does exactly channel average")
  2. Linear: target ~ log1p(trailing_avg_views), fit via simple regression

Usage:
    python -m models.baseline
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

from features.target import invert_target
from models.train import load_data, time_based_split


def main():
    df, _, _ = load_data()  # image/text embeddings loaded but unused here -- baseline is tabular-only
    train_idx, val_idx, test_idx = time_based_split(df)

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    X_train = np.log1p(train_df["trailing_avg_views"].values).reshape(-1, 1)
    y_train = train_df["target"].values
    X_test = np.log1p(test_df["trailing_avg_views"].values).reshape(-1, 1)
    y_test = test_df["target"].values

    # --- Naive baseline: always predict "video matches channel average" ---
    naive_preds = np.zeros_like(y_test)
    naive_rmse_target = np.sqrt(np.mean((naive_preds - y_test) ** 2))
    # Spearman is undefined for a constant prediction (zero variance) -- report as N/A, not 0
    print("--- Naive baseline (predict target=0 for everyone) ---")
    print(f"RMSE (target scale): {naive_rmse_target:.4f}")
    print("Spearman: N/A (constant prediction has no variance to correlate)\n")

    # --- Linear baseline: target ~ log1p(trailing_avg_views) ---
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    spearman_corr, _ = spearmanr(preds, y_test)
    rmse_target = np.sqrt(np.mean((preds - y_test) ** 2))

    predicted_views = invert_target(preds, test_df["trailing_avg_views"].values)
    actual_views = test_df["views"].values
    rmse_views = np.sqrt(np.mean((predicted_views - actual_views) ** 2))

    print("--- Linear baseline (target ~ log1p(trailing_avg_views)) ---")
    print(f"Learned coefficient: {model.coef_[0]:.4f}, intercept: {model.intercept_:.4f}")
    print(f"RMSE (target scale): {rmse_target:.4f}")
    print(f"Spearman correlation: {spearman_corr:.4f}")
    print(f"RMSE (original view-count space): {rmse_views:,.0f}")
    print("\nCompare these numbers directly against your fusion model's test results --")
    print("if the fusion model's Spearman isn't meaningfully higher than this, the")
    print("image/text branches aren't earning their complexity yet.")


if __name__ == "__main__":
    main()