"""
Diagnostics for the late-fusion regression run:
  1. Target distribution + skew (is compute_target() heavy-tailed?)
  2. Train/val/test distribution shift under the time-based split
  3. A cheap baseline (Ridge + LightGBM) on tabular + mean-pooled embeddings,
     scored with the same Spearman metric, to see if the late-fusion model
     is actually earning its complexity.

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
    print("High |skew| (>1) suggests a log/clip transform may help before "
          "feeding this into HuberLoss.\n")


def inspect_split_shift(df, train_idx, val_idx, test_idx):
    print("=== Target distribution by split (time-based) ===")
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        t = df.iloc[idx]["target"].values
        d = pd.to_datetime(df.iloc[idx]["published_at"])
        print(f"{name:5s}: n={len(idx):5d}  mean={t.mean():.4f}  median={np.median(t):.4f}  "
              f"std={t.std():.4f}  date_range=[{d.min().date()} -> {d.max().date()}]")
    print("Large mean/std swings across splits point to distribution shift, not just overfitting.\n")


def inspect_view_scale_shift(df, train_idx, val_idx, test_idx):
    print("=== Absolute view-count scale by split (checks for non-stationarity) ===")
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        v = df.iloc[idx]["views"].values
        tav = df.iloc[idx]["trailing_avg_views"].values
        print(f"{name:5s}: views      mean={v.mean():,.0f}  median={np.median(v):,.0f}  max={v.max():,.0f}")
        print(f"       trailing_avg mean={tav.mean():,.0f}  median={np.median(tav):,.0f}  max={tav.max():,.0f}")
    print("If test's mean/max are far above train's, RMSE growth is largely a scale-shift artifact, "
          "not model degradation -- consider reporting MAE/median-AE or log-space RMSE alongside it.\n")


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
    print(f"Top {top_n} rows account for {top_share:.1%} of total squared error "
          f"(out of {len(test_idx)} test rows). High concentration -> RMSE is outlier-driven, "
          f"not representative of typical error.")
    if is_first is not None:
        share_first = is_first[order].astype(bool).mean() if hasattr(is_first, "astype") else None
        overall_first_rate = pd.Series(test_sub["is_first_video"]).astype(bool).mean()
        print(f"is_first_video rate among top errors: {share_first:.1%}  "
              f"(vs {overall_first_rate:.1%} overall in test) -- if much higher, "
              f"first-video trailing_avg_views handling is likely distorting the target for these rows.\n")
    else:
        print()


def run_tabular_only_baseline(df, train_idx, test_idx, lgbm_with_embed_rho=None):
    print("=== Baseline: tabular features ONLY (no embeddings at all) ===")
    genre_categories = sorted(df.iloc[train_idx]["genre"].dropna().unique().tolist())
    train_tab, scaler = build_tabular_matrix(df.iloc[train_idx], genre_categories, fit_scaler=True)
    test_tab, _ = build_tabular_matrix(df.iloc[test_idx], genre_categories, scaler=scaler)
    y_train = df.iloc[train_idx]["target"].values
    y_test = df.iloc[test_idx]["target"].values

    if HAS_LGBM:
        lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.03, verbose=-1)
        lgbm.fit(train_tab, y_train)
        pred = lgbm.predict(test_tab)
        rho, _ = spearmanr(pred, y_test)
        print(f"LightGBM (tabular-only) Spearman: {rho:.4f}")
        importances = sorted(zip(lgbm.feature_importances_, range(train_tab.shape[1])), reverse=True)[:8]
        print(f"Top feature indices by importance: {[idx for _, idx in importances]}")
        if lgbm_with_embed_rho is not None:
            print(f"Compare this to LightGBM WITH full embeddings ({lgbm_with_embed_rho:.4f}) above:")
            print(f"  - if tabular-only ({rho:.4f}) is close -> embeddings add ~nothing, even unpooled")
            print(f"  - if tabular-only is much lower -> embeddings ARE carrying real signal, "
                  "and the late-fusion net is failing to extract it, not the embeddings' fault\n")
        else:
            print()
    else:
        print("LightGBM not installed -- skipping\n")


def run_baseline(df, train_idx, val_idx, test_idx, image_embeddings, text_embeddings):
    print("=== Baseline: tabular + FULL embeddings (all dims, not pooled) ===")
    # NOTE: a previous version of this baseline pooled each embedding down to
    # a single scalar via `.mean(axis=1)`, which destroys nearly all the
    # structure in the embedding space before the baseline ever sees it.
    # That made "tabular-only ~= tabular+embeddings" an artifact of the crude
    # pooling, not evidence that the embeddings lack signal. Ridge and
    # LightGBM can both consume the full-dimensional embeddings directly, so
    # we pass them through unpooled -- this is the fair version of the test.
    genre_categories = sorted(df.iloc[train_idx]["genre"].dropna().unique().tolist())
    train_tab, scaler = build_tabular_matrix(df.iloc[train_idx], genre_categories, fit_scaler=True)
    test_tab, _ = build_tabular_matrix(df.iloc[test_idx], genre_categories, scaler=scaler)

    def full_embed(idx):
        return np.hstack([image_embeddings[idx], text_embeddings[idx]])

    X_train = np.hstack([train_tab, full_embed(train_idx)])
    X_test = np.hstack([test_tab, full_embed(test_idx)])
    y_train = df.iloc[train_idx]["target"].values
    y_test = df.iloc[test_idx]["target"].values

    xs = StandardScaler().fit(X_train)
    X_train_s, X_test_s = xs.transform(X_train), xs.transform(X_test)

    # Ridge with ~800+ correlated embedding dims and a few thousand rows is
    # prone to overfitting even with alpha=1.0 -- included mainly as a linear
    # reference point, LightGBM is the more trustworthy baseline here.
    ridge = Ridge(alpha=10.0).fit(X_train_s, y_train)
    ridge_pred = ridge.predict(X_test_s)
    ridge_rho, _ = spearmanr(ridge_pred, y_test)
    print(f"Ridge         Spearman: {ridge_rho:.4f}")

    lgbm_rho = None
    if HAS_LGBM:
        lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.03, verbose=-1)
        lgbm.fit(X_train, y_train)
        lgbm_pred = lgbm.predict(X_test)
        lgbm_rho, _ = spearmanr(lgbm_pred, y_test)
        print(f"LightGBM      Spearman: {lgbm_rho:.4f}")
    else:
        print("LightGBM not installed -- skipping (pip install lightgbm to include it)")

    print("Compare both against the late-fusion model's test Spearman (see below).")
    print("If either baseline matches or beats it, the net isn't extracting the signal")
    print("that's actually available in the embeddings -- an optimization/training")
    print("problem in the net, not evidence the embeddings themselves are uninformative.\n")
    return lgbm_rho


def load_trained_model_test_predictions(df, test_idx, image_embeddings, text_embeddings):
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
    _, preds, targets = evaluate(model, test_loader, torch.nn.HuberLoss(), device)
    test_spearman, _ = spearmanr(preds, targets)

    predicted_views = invert_target(preds, test_sub["trailing_avg_views"].values)
    actual_views = test_sub["views"].values
    return predicted_views, actual_views, test_spearman


def main():
    df, image_embeddings, text_embeddings = load_data()
    train_idx, val_idx, test_idx = time_based_split(df)

    inspect_target_distribution(df)
    inspect_split_shift(df, train_idx, val_idx, test_idx)
    inspect_view_scale_shift(df, train_idx, val_idx, test_idx)
    lgbm_with_embed_rho = run_baseline(df, train_idx, val_idx, test_idx, image_embeddings, text_embeddings)
    run_tabular_only_baseline(df, train_idx, test_idx, lgbm_with_embed_rho)

    predicted_views, actual_views, test_spearman = load_trained_model_test_predictions(
        df, test_idx, image_embeddings, text_embeddings
    )
    print(f"Late-fusion net test Spearman: {test_spearman:.4f}\n")
    inspect_top_test_errors(df, test_idx, predicted_views, actual_views)


if __name__ == "__main__":
    main()