# yt-performance-predictor

Predicts how a YouTube video will perform (views, relative to the channel's own baseline) from its **thumbnail + title alone, before it's published**, and is being built out into a continuously retrained, deployed service.

## Status

| Area | State |
|---|---|
| Ingestion (YouTube API, thumbnails, title features) | Built, runs locally, writes `data/videos.csv` |
| Embedding + visual-feature caches | Built (`models/precompute_*.py`) |
| Late-fusion model + training + experiment log | Built; ~10.6k finalized rows, test Spearman ≈ 0.33 (see [Results](#results)) |
| Baselines / diagnostics | Built (`models/baseline.py`, `models/inspect_target_and_baseline.py`, `models/ablation_modalities.py`) |
| Model bundle export, serving code, Docker images | **Not built yet** |
| Postgres, Airflow, retrain/promotion, drift monitoring | **Not built yet** (designed, see [Deployment plan](#deployment-plan-planned-not-yet-implemented)) |
| Deployment (Oracle Cloud VM) | **Not started** |

Everything under "Deployment plan" and "Roadmap" is a design, not running code.

## What it predicts

Given a **thumbnail image + title** (known before publish), predict the video's **relative performance**: how it will do compared to that channel's own recent average, not raw view count. Raw views are dominated by channel size; normalizing against the channel's own baseline isolates the effect of the thumbnail/title itself.

```
target = log(1 + views) − log(1 + trailing_avg_views)
```

`trailing_avg_views` is the channel's mean views over its 5 most recent prior uploads (`features/trailing_views.py`, rolling window shifted by one so a video never sees itself). Predicted views are recovered with `invert_target`: `expm1(score + log1p(trailing_avg_views))`.

## Pipeline today (local)

```
YouTube Data API v3 ──► pipeline/run_ingestion.py ──► data/videos.csv + data/images/
                                                          │
                          ┌───────────────────────────────┴───────────────┐
                          ▼                                               ▼
        models/precompute_embeddings.py                 models/precompute_visual_features.py
        (frozen CLIP / DINOv2 / MiniLM,                 (faces, text overlay, color stats
         cached to data/embeddings/<encoder>/)           → data/visual_features.csv)
                          └───────────────────────────────┬───────────────┘
                                                          ▼
                                                  models/train.py
                                   (late-fusion head, chronological split,
                                    appends each run to experiments/results.jsonl)
```

## Data

- **Source:** YouTube Data API v3 (`playlistItems.list` + `videos.list`; `search.list` is deliberately avoided, it costs ~100x more quota).
- **Scope:** hand-curated English-speaking channels, excluding channels whose views are driven by external events (news, official music channels tied to release calendars). Personality, gaming, commentary and entertainment channels are the target profile. `pipeline/config/channels.json` is the list `run_ingestion` reads (currently 3 handles); `pipeline/config/cum_channels.json` holds a larger list of 143 handles.
- **Window:** videos published from **2025-01-01** onward.
- **Shorts excluded:** anything ≤180s is dropped before download or storage.
- **Size:** the latest experiments train on 8,472 rows, validate on 1,059 and test on 1,060 (10,591 usable rows after filtering).
- **Storage (local for now):** `data/images/{video_id}.jpg`, `data/videos.csv`, `data/visual_features.csv`, `data/embeddings/<image_encoder>/`. Raw images are kept as the source of truth so encoders can be swapped without re-downloading.

### Fields retrieved vs. derived

| Retrieved (API) | Derived locally |
|---|---|
| Thumbnail image | Title length, word count, capitalized-word/letter counts and ratio, symbol count, has-question-mark, has-number |
| Title (raw text) | Trailing average views (rolling, shifted), `is_first_video` |
| Views, duration, publish date | Face count / has-face, text-overlay flag (MTCNN, EasyOCR) |
| Subscriber count | Thumbnail color stats: mean saturation, brightness, brightness std, warm-hue ratio |
| `categoryId` → genre | Optional `clip_sim`: cosine similarity between CLIP image and CLIP title embeddings |

## Model: v1 late fusion

Three branches, each with its own frozen pretrained encoder (where applicable), a small learned projection, then concatenation and an MLP regression head.

- **Image branch:** default **CLIP ViT-B/32** (512-d, thumbnails squashed to 224x224). Alternatives selectable at embedding time: DINOv2 ViT-S/14 (384-d), CLIP ViT-B/16.
- **Text branch:** default **CLIP text tower** on the title (512-d), same space as the CLIP image embedding. Alternative: `all-MiniLM-L6-v2` (384-d).
- **Tabular branch:** log-scaled subscriber count and trailing average views, duration, title stats, face count, color stats, boolean flags (question mark, number, has-face, text overlay), one-hot genre, and optionally `clip_sim`.
- **Fusion:** each branch → 32-d projection (ReLU + dropout) → concatenate → 64 → 16 → 1.
- **Encoders are fully frozen.** Only projections and the head train. Embeddings are computed once and cached, since frozen encoders give identical outputs every epoch.

**Training setup** (`models/train.py`): Huber loss, Adam (lr 2e-5, weight decay 1e-4), batch 64, dropout 0.2, Gaussian noise (std 0.02) on the embeddings during training, early stopping on validation loss (patience 10, max 200 epochs), best-val-loss checkpoint kept. Split is **chronological by `published_at`** at 80/10/10 by row fraction, so validation and test are always the newest videos. Scalers are fit on the training split only.

**Metrics:** Spearman rank correlation (the honest metric for "will thumbnail A beat thumbnail B"), AUC for over- vs. under-performing the channel baseline, and error in target space (plus RMSE/MAE in view space via `invert_target`).

### Results

Mean ± std over seeds on the same 1,060-row test split (`experiments/results.jsonl`, 23 runs):

| Text encoder | `clip_sim` | Seeds | Test Spearman | Test AUC | Test MAE (target) |
|---|---|---|---|---|---|
| CLIP | off | 6 | 0.335 ± 0.015 | 0.644 | 0.490 |
| CLIP | on | 6 | 0.333 ± 0.016 | 0.640 | 0.488 |
| MiniLM | off | 6 | 0.321 ± 0.012 | 0.641 | 0.484 |
| MiniLM | on | 5 | 0.327 ± 0.003 | 0.643 | 0.486 |

All rows use the CLIP ViT-B/32 image encoder. Differences between these configurations are within seed-to-seed noise, so there is no evidence yet that one text encoder or the similarity feature is better. The signal is real but modest. `models/baseline.py` and `models/inspect_target_and_baseline.py` compare against naive, linear, Ridge and LightGBM baselines on the identical split; run them to see how much of the score comes from the image and text branches rather than from `trailing_avg_views` alone.

## Known limitations (tracked, not hidden)

- **`subscriber_count_at_upload` is the current subscriber count**, not the value at publish time. `ingestion/tubecensus_client.py` is a stub that returns the fallback; the TubeCensus integration was shelved (large local storage requirement, Windows permission issues). The deployment plan fixes this going forward by snapshotting channel state at ingest time. Old rows keep the approximation.
- **Label timing is not a fixed horizon.** Today `label_finalized` flips once a video is ≥28 days old *at the time of an ingestion run*, and views are whatever the API returns at that moment. For backfilled videos that can be many months, not day 28. The planned redesign ingests right after publish and reads views once at day 28.
- **Trailing baseline in incremental runs (found in code review, verify before relying on it).** `compute_trailing_views` runs over only the videos fetched in the current run, and already-finalized videos are excluded from that fetch. On re-runs, the baseline for newly fetched videos may therefore be computed from a truncated history. Computing it from the full per-channel history (e.g. in Postgres) is part of the redesign.
- **`is_first_video`** means the first video in our fetched 2025+ window for that channel, not the channel's first upload ever.
- **Visual features are heavy and their value is unmeasured in the committed results.** MTCNN and EasyOCR dominate the size of any serving image. `models/ablation_modalities.py` is meant to test their contribution; results are not committed.
- **Checkpoint format is not serving-ready.** `train.py` saves weights plus a pickled sklearn scaler tuple, but not the feature column order (implied by module-level lists in `models/dataset.py`, one of which `load_data()` mutates when `USE_SIM=1`). Serving needs a versioned bundle (weights, scalers, column order, encoder names) and a parity test against the training path.
- **Delayed labels:** rows are only used for training when `label_finalized` is true, and finalized rows are never re-fetched.
- Ingestion re-runs are idempotent and incremental: thumbnails are skipped if already downloaded, and finalized `video_id`s are excluded from the `videos.list` fetch.

## Deployment plan (planned, not yet implemented)

### Infrastructure

- **One Oracle Cloud Always Free Ampere A1 VM (2 OCPU / 12 GB, arm64)** running Docker Compose: Caddy (HTTPS), FastAPI, Postgres, Airflow, and a worker image. Oracle halved the Always Free A1 allowance from 4 OCPU / 24 GB to 2 OCPU / 12 GB in June 2026, so the design targets the smaller size. Idle Always Free instances can be reclaimed (CPU, network and memory all under 20% for 7 days), and the free tier is not a guarantee, so **everything must be rebuildable and backed up off the VM.**
- Airflow decides when and in what order jobs run. Worker containers do the work (`DockerOperator`), so torch/pandas pins don't conflict with Airflow's.
- No AWS unless training v2 needs a GPU spot instance (Airflow launches it, S3 for artifacts).
- Keep the compose file portable: the same file should run on EC2 if the free tier changes again.
- Build `linux/arm64` images (buildx, or build on the VM).

### Storage

- **Postgres:** two databases, `airflow` and app data (videos, features, run logs, model registry). Replaces `videos.csv`.
- **Block volume:** thumbnails, embeddings, checkpoints, model caches. Nightly `pg_dump` and checkpoint backup off the VM.

### DAGs

| DAG | When | Does |
|---|---|---|
| `ingest_new` | daily | videos not in DB yet: thumbnail, title, channel snapshot (subs, baseline) at ingest time |
| `embed_new` | after ingest | encoder embeddings and visual features for rows missing them (idempotent) |
| `finalize_labels` | daily | read views at ~28 days, set `label_finalized`, never rewrite |
| `retrain` | periodic or on drift | build dataset, train challenger, gate, compare, promote or reject |
| `monitor_drift` | weekly | error on newly finalized videos plus feature shift; can trigger `retrain` |
| `backup` | nightly | `pg_dump` and checkpoints off the VM |

Rules: idempotent tasks, heavy tasks in a pool of size 1, `max_active_runs=1`, retries with backoff, log every run.

### Retrain and promotion

- Chronological split by `published_at`. The test slice is a fixed recent time window (not a fraction), so champion and challenger score on the same rows, and it must be newer than anything the champion trained on (`train_end` stored in the registry).
- Re-score the champion on that slice every time; never compare against its stored old metrics.
- Skip the run if there aren't enough new finalized rows.
- **Hard gate:** no NaNs, checkpoint loads with the right dimensions, prediction spread not ~0, beats the constant-mean baseline, Spearman > 0.
- **Soft comparison:** paired bootstrap 95% CI of ΔSpearman (challenger − champion) on shared rows. Reject only if the whole CI is below 0; otherwise promote, and ties go to the newer model. Small slices are noisy (Spearman on ~200 rows is roughly ±0.07), so "must beat the champion" would freeze the model on stale data.
- Keep the previous champion for rollback and log every candidate's metrics.

### Inference

- FastAPI takes thumbnail + title + channel, computes the same features as training, and returns the predicted score, expected views and model version.
- Channel baseline and subscriber count come from Postgres; for unseen channels the caller supplies them.
- **Model bundle** = weights + fitted scalers + feature column order, versioned together. Encoder weights baked into the image. Champion loaded at startup and reloaded on promotion.
- Rate-limit `/predict`.

### Security

Airflow UI behind auth. Postgres never public. Secrets in `.env`, not in images. SSH key-only.

## Roadmap

1. Export a versioned model bundle from `train.py` and add a train/serve parity test.
2. Slim serving requirements and an arm64 Dockerfile; deploy Caddy + FastAPI to the Oracle VM.
3. Move `videos.csv` into Postgres; turn ingest, embed and finalize into containerized CLI commands.
4. Wrap those commands in Airflow DAGs; add `backup`.
5. Add `retrain` with the promotion logic above, then `monitor_drift`.
6. **v2, early fusion:** cross-attention transformer over image patch tokens and title tokens jointly, to capture thumbnail/title mismatch signals late fusion can't see. To be compared against v1 on the same data; adopted only if measurably better.

### Open decisions

- Face/OCR features at serving (MTCNN + EasyOCR are heavy): keep, replace with lighter versions, or drop, based on the ablation.
- Serve CLIP only (image + text towers) and drop MiniLM/DINOv2 from the serving image, since the encoder differences above are within noise.
- Retrain cadence, minimum new rows, and test-slice length.
- Rejection rule: strictly "CI below 0", or with a small margin.
- v2 training: local GPU or AWS spot.

## Repo structure

```
ingestion/     YouTube API client, thumbnail downloader, subscriber lookup (TubeCensus stub)
features/      Title features, trailing views, target, visual features (faces, OCR, color)
pipeline/      run_ingestion.py and config/ (channels.json, cum_channels.json)
models/
  precompute_embeddings.py        frozen image/text embeddings, cached
  precompute_visual_features.py   face / text-overlay / color features, cached
  dataset.py                      tabular matrix assembly + torch Dataset
  late_fusion_model.py            LateFusionModel (+ standalone shape test)
  train.py                        training, evaluation, experiment logging
  baseline.py                     naive / linear / fusion comparison
  inspect_target_and_baseline.py  target diagnostics, Ridge / LightGBM comparison
  ablation_modalities.py          tabular / text / image / full-fusion ablation
experiments/   results.jsonl (one line per training run)
data/          local dataset output (gitignored: images/, videos.csv, embeddings/, ...)
current_EDA.ipynb
```

## Setup

```bash
pip install -r requirements-train.txt
```

`requirements-train.txt` is a full training/dev freeze: it pins CUDA builds of torch (`+cu128`), Jupyter, and TubeCensus-related packages. A slim serving requirements file is planned and is part of the deployment work.

Set in `.env` (loaded via `python-dotenv`, must load before any project imports):
```
YOUTUBE_API_KEY=your_key_here
```

## Running it

```bash
# 1. Ingest data (incremental; safe to re-run)
python -m pipeline.run_ingestion

# 2. Sanity-check the model architecture (no data or pretrained weights needed)
python models/late_fusion_model.py

# 3. Precompute frozen embeddings (default: CLIP ViT-B/32 image, MiniLM + CLIP text)
python -m models.precompute_embeddings
#    other options: --image-encoder dinov2|clip_b16, --text-encoder minilm|clip|both,
#                   --image-mode squash|crop, --limit N (smoke test)

# 4. Precompute visual features (has_face, text overlay, color stats)
python -m models.precompute_visual_features

# 5. Train (env vars: IMAGE_ENCODER, TEXT_ENCODER, USE_SIM, SEED)
python -m models.train
IMAGE_ENCODER=clip_b32 TEXT_ENCODER=minilm USE_SIM=1 SEED=3 python -m models.train

# 6. Baselines, diagnostics and ablations
python -m models.baseline
python -m models.inspect_target_and_baseline
python -m models.ablation_modalities
```

Each training run appends its config and test metrics to `experiments/results.jsonl` and saves the best checkpoint to `models/checkpoints/late_fusion_v1_<image_encoder>[_<text_encoder>][_sim].pt`.