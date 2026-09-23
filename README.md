# yt-performance-predictor

Predicts how a YouTube video will perform (views, relative to the channel's own baseline) from its **thumbnail + title alone, before it's published** — with a continuous training pipeline that keeps the model current as new data arrives.

## Why this exists

Most "will this thumbnail work" tools are black boxes. This project builds one end-to-end: real data ingestion, honest handling of the messy parts (delayed labels, channel-size normalization, staleness), a multimodal deep learning model, and a plan to run it as a continuously retrained production system rather than a one-off notebook.

## What it predicts

Given a **thumbnail image + title** (known before publish), predict the video's **relative performance** — how it will do compared to that channel's own recent average, not raw view count. Raw views are dominated by channel size; normalizing against the channel's own baseline isolates the effect of the thumbnail/title itself.

```
target = log(1 + views) − log(1 + trailing_avg_views)
```

Where `trailing_avg_views` is the channel's average views over its N most recent prior uploads — computed causally (only videos published *before* the one in question), so it's genuinely available at prediction time with no leakage.

## Pipeline overview

```
YouTube Data API v3  ──►  ingestion/  ──►  data/videos.csv + data/images/
                                              │
                                              ▼
                                    features/ (title stats, trailing views)
                                              │
                                              ▼
                              models/precompute_embeddings.py
                              (frozen DINOv2 + MiniLM, cached once)
                                              │
                                              ▼
                                     models/train.py
                              (late fusion head, time-based split)
```

## Data

- **Source:** YouTube Data API v3 (`playlistItems.list` + `videos.list`, cheap endpoints — avoid `search.list`, which costs 100x more quota per call).
- **Scope:** ~100 English-speaking channels, hand-curated to exclude channels where views are driven by external events rather than thumbnail/title choice (news channels, official music/artist channels tied to release calendars). Personality, gaming, commentary, and entertainment channels are the target profile.
- **Window:** videos published from **2025-01-01 onward**.
- **Shorts excluded:** anything ≤180s duration is filtered out before download/storage — Shorts are discovered completely differently (feed-driven) and don't share the same performance dynamics as long-form video.
- **Storage:** local disk for now (`data/images/{video_id}.jpg`, `data/videos.csv`). Raw images are kept as source of truth rather than only storing embeddings, so encoders can be swapped or fine-tuned later without re-downloading.

### Fields retrieved vs. derived

| Retrieved (API) | Derived (computed locally) |
|---|---|
| Thumbnail image | Title length, word count |
| Title (raw text) | Capitalized-word count, capitalized-letter count & ratio |
| Views | Symbol count, has-question-mark, has-number |
| Subscriber count | Trailing average views (rolling, shifted to prevent leakage) |
| `categoryId` → genre | `is_first_video` flag (no prior videos in window) |
| Duration, publish date | |

### Known limitations (tracked, not hidden)

- **`subscriber_count_at_upload` is actually current subscriber count**, not the true value at publish time. A TubeCensus integration (historical subscriber snapshots via Wayback Machine data) was attempted and shelved — real package, but required ~20GB local storage and hit Windows permission issues with diminishing returns for a v1. Revisit later if this proves to matter; a coarse tier bucket (e.g. `<100K / 100K-1M / 1M-5M / ...`) is a cheap partial fix if needed, since it's far less sensitive to staleness than an exact number.
- **Delayed labeling:** a video's view count isn't "final" the moment it's published. Rows are only used for training once `label_finalized = True`, which flips after ~28 days. Views are refreshed on a decreasing cadence as a video ages (always for <2 months old, checked for >5% change for 2-6 months, never touched beyond 6 months) — but once `label_finalized` flips, that row's views are never rewritten again, ever, for reproducibility.
- **`is_first_video`** means "first video *in our fetched 2025+ window*" for that channel, not literally the channel's first upload ever — there's no trailing-views signal available for these regardless of channel age, since earlier history wasn't pulled.
- Ingestion re-runs are **idempotent and incremental**: thumbnails are skipped if already downloaded, and already-finalized `video_id`s are excluded from the `videos.list` fetch entirely (not just discarded after fetching), so quota cost scales with what's actually new, not total dataset size.

## Model — v1: Late Fusion

Three independent branches, each with its own frozen pretrained encoder, projected and concatenated before a small regression head:

- **Image branch:** DINOv2 ViT-S/14 (self-supervised, 384-dim) — chosen over a supervised ImageNet backbone (e.g. ConvNeXt) because supervised features are compressed around 1,000-category object discrimination, discarding compositional/aesthetic signal (color, contrast, framing) that plausibly matters more for thumbnail performance than "what object is this."
- **Text branch:** `all-MiniLM-L6-v2` sentence embedding (384-dim) on the title.
- **Tabular branch:** subscriber count (log-scaled), trailing average views (log-scaled), duration, title-derived stats, one-hot genre.
- **Fusion:** each branch → its own small projection layer → concatenated → MLP regression head → single scalar (the relative-performance target).

Both encoders are **fully frozen** in v1 — only the projections and fusion head are trained. Embeddings are precomputed once (`models/precompute_embeddings.py`) and cached, since frozen encoders produce the same output every epoch; re-running them repeatedly during training would be pure waste.

**Split:** time-based (train on earliest videos, validate/test on most recent finalized ones) — matches real deployment (predicting forward) and avoids random-split leakage across near-duplicate time windows.

**Loss:** Huber (robust to view-count outliers). **Early stopping** on validation loss (patience=5) to avoid training past the overfitting point — only the best checkpoint by val loss is kept.

**Evaluation metrics:**
- RMSE, converted back to raw view-count space via `invert_target`
- **Spearman rank correlation** between predicted and actual relative performance — arguably the more honest metric for this task, since the real use case ("will thumbnail A beat thumbnail B") is fundamentally a ranking question, not a point-estimate one.

## Planned — v2: Early Fusion

Once v1's pipeline is proven and enough data has accumulated, build a cross-attention transformer over image patch tokens + title tokens jointly (rather than separately-pooled embeddings), to capture thumbnail/title *mismatch* signals late fusion can't see (e.g. clickbait where the two don't agree). Compared empirically against v1 on the same accumulated dataset — architecture choice justified by measured improvement, not assumed.

## Planned — Continuous Training Infrastructure

- **Orchestration:** Airflow DAGs for scheduled ingestion, delayed-label refresh, and retraining triggers.
- **Compute:** Docker containers on AWS EC2 (spot instances for training cost control).
- **Drift monitoring:** track prediction error and feature distributions over time; YouTube's algorithm and audience behavior genuinely shift over months, giving real (not synthetic) retraining triggers.
- **Storage at scale:** S3 for images, Postgres (self-hosted on the same EC2 instance, or RDS free tier) for structured data — deliberately avoiding a second managed-service vendor (Supabase/Oracle/Neon were evaluated and dropped) since consolidating onto one cloud reduces operational surface area without a compelling technical reason to split it.
- **Serving:** FastAPI + Docker, with the encoders exported to ONNX and quantized (fp16/int8) for a lightweight, free-tier-friendly deployment footprint — same technique already used in a prior project (Speech2Market) to hit a memory-constrained deployment target.

## Repo structure

```
ingestion/          YouTube API client, thumbnail downloader, subscriber lookup
features/           Title feature engineering, trailing views, target computation
pipeline/           Orchestration script tying ingestion + features together
models/             Embedding precomputation, dataset, late fusion architecture, training
data/                Local dataset output (gitignored: images/, videos.csv, embeddings/)
```

## Setup

```bash
pip install -r requirements.txt
```

Set in `.env` (loaded via `python-dotenv`, must load before any project imports):
```
YOUTUBE_API_KEY=your_key_here
```

## Running it

```bash
# 1. Ingest data (incremental -- safe to re-run as CHANNELS grows)
python -m pipeline.run_ingestion

# 2. Sanity-check the model architecture (no data or pretrained weights needed)
python models/late_fusion_model.py

# 3. Precompute frozen embeddings (one-time, or after adding new data)
python -m models.precompute_embeddings

# 4. Precompute visual-related features (has_face, overlays, etc.)
python -m models.precompute_visual_features

# 5. Train
python -m models.train

# 6. Ablation tests
python -m models.ablation_tabular_only
python -m models.inspect_target_and_baseline
```

## Status

v1 late fusion model runs end-to-end on ~6,800 finalized rows. Actively iterating on regularization (dropout/weight decay) to address overfitting observed after ~epoch 7 on the first full run.