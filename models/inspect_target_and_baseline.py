"""
Diagnostics for the late-fusion regression run:
  1. Target distribution + skew (is compute_target() heavy-tailed?)
  2. Train/val/test distribution shift under the time-based split
  3. Model comparison table: Ridge and LightGBM (tabular + full embeddings),
     LightGBM (tabular-only), and the trained late-fusion net -- all scored
     with Spearman on the identical test split, in one table.
  4. Top-error inspection (is RMSE outlier-dominated?)

Usage:
    python -m models.inspect_target_and_baseline
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import skew, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from models.train import load_data, time_based_split, CHECKPOINT_PATH, evaluate
from models.dataset import build_tabular_matrix, VideoDataset
from models.late_fusion_model import LateFusionModel
from features.target import invert_target
from torch.utils.data import DataLoader

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def inspect_target_distribution(df):
    print("=== Target distribution ===")
    t = df["target"].values
    print(f"n={len(t)}  mean={t.mean():.4f}  median={np.median(t):.4f}  "
          f"std={t.std():.4f}  skew={skew(t):.4f}")
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"  p{p:>2}: {np.percentile(t, p):.4f}")
    n_extreme = ((t > np.percentile(t, 99)) | (t < np.percentile(t, 1))).sum()
    print(f"Rows outside 1st/99th pct: {n_extreme} ({n_extreme/len(t):.2%})")
    print("(High |skew| > 1 suggests a log/clip transform may help before HuberLoss.)\n")


def inspect_split_shift(df, train_idx, val_idx, test_idx):
    print("=== Target distribution by split (time-based) ===")
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        t = df.iloc[idx]["target"].values
        d = pd.to_datetime(df.iloc[idx]["published_at"])
        print(f"{name:5s}: n={len(idx):5d}  mean={t.mean():.4f}  median={np.median(t):.4f}  "
              f"std={t.std():.4f}  date_range=[{d.min().date()} -> {d.max().date()}]")
    print("(Large mean/std swings across splits point to distribution shift, not just overfitting.)\n")


def inspect_view_scale_shift(df, train_idx, val_idx, test_idx):
    print("=== Absolute view-count scale by split ===")
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        v = df.iloc[idx]["views"].values
        tav = df.iloc[idx]["trailing_avg_views"].values
        print(f"{name:5s}: views mean={v.mean():,.0f} median={np.median(v):,.0f} max={v.max():,.0f}  |  "
              f"trailing_avg mean={tav.mean():,.0f} median={np.median(tav):,.0f} max={tav.max():,.0f}")
    print("(If test's mean/max are far above train's, RMSE growth is a scale-shift artifact, not just model decay.)\n")


def get_tabular_only_predictions(df, train_idx, test_idx):
    genre_categories = sorted(df.iloc[train_idx]["genre"].dropna().unique().tolist())
    train_tab, scaler = build_tabular_matrix(df.iloc[train_idx], genre_categories, fit_scaler=True)
    test_tab, _ = build_tabular_matrix(df.iloc[test_idx], genre_categories, scaler=scaler)
    y_train = df.iloc[train_idx]["target"].values

    if not HAS_LGBM:
        return None, None
    lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.03, verbose=-1)
    lgbm.fit(train_tab, y_train)
    pred = lgbm.predict(test_tab)
    top_features = sorted(zip(lgbm.feature_importances_, range(train_tab.shape[1])), reverse=True)[:8]
    return pred, [idx for _, idx in top_features]


def get_embedding_baseline_predictions(df, train_idx, test_idx, image_embeddings, text_embeddings):
    """Ridge and LightGBM on tabular + FULL (unpooled) embeddings. Passing the
    full-dimensional embeddings, not a crude mean-pooled scalar, is what makes
    this a fair test of whether the embeddings carry real signal."""
    genre_categories = sorted(df.iloc[train_idx]["genre"].dropna().unique().tolist())
    train_tab, scaler = build_tabular_matrix(df.iloc[train_idx], genre_categories, fit_scaler=True)
    test_tab, _ = build_tabular_matrix(df.iloc[test_idx], genre_categories, scaler=scaler)

    def full_embed(idx):
        return np.hstack([image_embeddings[idx], text_embeddings[idx]])

    X_train = np.hstack([train_tab, full_embed(train_idx)])
    X_test = np.hstack([test_tab, full_embed(test_idx)])
    y_train = df.iloc[train_idx]["target"].values

    xs = StandardScaler().fit(X_train)
    X_train_s, X_test_s = xs.transform(X_train), xs.transform(X_test)

    # Ridge with ~800+ correlated embedding dims and a few thousand rows is
    # prone to overfitting even with alpha=10 -- included as a linear
    # reference point only, LightGBM is the more trustworthy of the two.
    ridge_pred = Ridge(alpha=10.0).fit(X_train_s, y_train).predict(X_test_s)

    lgbm_pred = None
    if HAS_LGBM:
        lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.03, verbose=-1)
        lgbm.fit(X_train, y_train)
        lgbm_pred = lgbm.predict(X_test)

    return ridge_pred, lgbm_pred


def get_fusion_model_predictions(df, test_idx, image_embeddings, text_embeddings):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    model = LateFusionModel(
        image_dim=checkpoint["image_dim"],
        text_dim=checkpoint["text_dim"],
        tabular_dim=checkpoint["tabular_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_tabular, _ = build_tabular_matrix(df.iloc[test_idx], checkpoint["genre_categories"], scaler=checkpoint["scaler"])
    test_sub = df.iloc[test_idx]
    test_ds = VideoDataset(
        image_embeddings[test_idx], text_embeddings[test_idx], test_tabular,
        test_sub["target"].values, test_sub["video_id"].values,
    )
    test_loader = DataLoader(test_ds, batch_size=512)
    _, preds, _ = evaluate(model, test_loader, torch.nn.HuberLoss(), device)
    return preds, checkpoint


def print_model_comparison_table(y_test, predictions):
    """predictions: dict of {model_name: preds_array or None}. Builds one
    Spearman-vs-Spearman table so every model's ranking skill is visible
    side by side, instead of scattered across separate print blocks."""
    rows = {}
    for name, pred in predictions.items():
        if pred is None:
            rows[name] = float("nan")
        else:
            rho, _ = spearmanr(pred, y_test)
            rows[name] = rho

    table = pd.Series(rows, name="Spearman").sort_values(ascending=False)
    print("=== Model comparison: Spearman on identical test split (higher = better ranking) ===\n")
    print(table.to_string(float_format=lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A (not installed)"))

    print("\nHow to read this:")
    print("- Ranked highest to lowest -- the late-fusion net should beat every tabular-only")
    print("  baseline if the image/text branches are earning their complexity.")
    print("- If 'LightGBM (tabular + embeddings)' or 'Ridge (tabular + embeddings)' beats the")
    print("  late-fusion net, the embeddings DO carry usable signal -- the net is failing to")
    print("  extract it (an optimization/architecture problem), not evidence the embeddings")
    print("  themselves are uninformative.")
    print("- If 'LightGBM (tabular-only)' is close to 'LightGBM (tabular + embeddings)', the")
    print("  embeddings add ~nothing even to a model that CAN use them well.\n")


def inspect_top_test_errors(df, test_idx, predicted_views, actual_views, top_n=10):
    print(f"=== Top {top_n} test errors by squared error (checks RMSE outlier dominance) ===")
    sq_err = (predicted_views - actual_views) ** 2
    order = np.argsort(sq_err)[::-1][:top_n]
    total_sq_err = sq_err.sum()
    top_share = sq_err[order].sum() / total_sq_err
    test_sub = df.iloc[test_idx]
    video_ids = test_sub["video_id"].values
    is_first = test_sub["is_first_video"].values if "is_first_video" in test_sub.columns else None
    tav = test_sub["trailing_avg_views"].values

    for i in order:
        first_flag = f"  is_first_video={is_first[i]}" if is_first is not None else ""
        print(f"  video_id={video_ids[i]}  actual={actual_views[i]:,.0f}  "
              f"predicted={predicted_views[i]:,.0f}  trailing_avg={tav[i]:,.0f}  "
              f"sq_err={sq_err[i]:,.0f}{first_flag}")

    print(f"\nTop {top_n} rows account for {top_share:.1%} of total squared error "
          f"(out of {len(test_idx)} test rows).")
    print("(High concentration -> RMSE is outlier-driven, not representative of typical error.)")

    if is_first is not None:
        share_first = pd.Series(is_first[order]).astype(bool).mean()
        overall_first_rate = pd.Series(test_sub["is_first_video"]).astype(bool).mean()
        print(f"is_first_video rate among top errors: {share_first:.1%} vs {overall_first_rate:.1%} overall "
              f"in test.")
        if share_first > overall_first_rate * 1.5:
            print("-> Notably higher: first-video trailing_avg_views handling is likely distorting "
                  "the target for these rows.")
    print()


def main():
    df, image_embeddings, text_embeddings = load_data()
    train_idx, val_idx, test_idx = time_based_split(df)
    y_test = df.iloc[test_idx]["target"].values

    inspect_target_distribution(df)
    inspect_split_shift(df, train_idx, val_idx, test_idx)
    inspect_view_scale_shift(df, train_idx, val_idx, test_idx)

    ridge_pred, lgbm_embed_pred = get_embedding_baseline_predictions(
        df, train_idx, test_idx, image_embeddings, text_embeddings
    )
    lgbm_tabular_pred, top_features = get_tabular_only_predictions(df, train_idx, test_idx)
    fusion_pred, checkpoint = get_fusion_model_predictions(df, test_idx, image_embeddings, text_embeddings)

    print_model_comparison_table(y_test, {
        "Late-fusion net": fusion_pred,
        "LightGBM (tabular + embeddings)": lgbm_embed_pred,
        "Ridge (tabular + embeddings)": ridge_pred,
        "LightGBM (tabular-only)": lgbm_tabular_pred,
    })

    if top_features is not None:
        print(f"LightGBM (tabular-only) top feature indices by importance: {top_features}\n")

    print(f"Fusion model checkpoint: epoch {checkpoint.get('epoch', '?')}, "
          f"val_loss={checkpoint.get('val_loss', float('nan')):.4f}\n")

    test_sub = df.iloc[test_idx]
    predicted_views = invert_target(fusion_pred, test_sub["trailing_avg_views"].values)
    actual_views = test_sub["views"].values
    inspect_top_test_errors(df, test_idx, predicted_views, actual_views)


if __name__ == "__main__":
    main()