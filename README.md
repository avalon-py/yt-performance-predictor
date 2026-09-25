# yt-performance-predictor

Predicts a YouTube video's relative performance (views vs. the channel's own recent baseline) from **thumbnail + title alone, before publish**. Built as a continuously-retrained, deployed service.

## Status

| Area | State |
|---|---|
| Ingestion, embeddings, late-fusion model + training | Built. ~10.6k rows, test Spearman ≈ 0.33 (see [Results](#results)) |
| Model bundle export + train/serve parity test | Built and passing (0/200 samples outside tolerance) |
| Serving (FastAPI, Dockerfile, requirements-serve.txt) | Built and validated locally (amd64) — containerized CPU predictions match the training path exactly |
| Docker Compose stack (api + Caddy) | Built and validated locally — both containers build and start cleanly, api healthcheck passes (`healthy`), `/health` confirmed reachable from Caddy over the internal Compose network |
| Deployment (Oracle Cloud VM, Caddy, arm64) | Blocked on Oracle Always Free Ampere A1 capacity; `docker-compose.yml` + `Caddyfile` written and locally validated on amd64, not yet live on arm64 |
| Postgres, Airflow, retrain/promotion, drift monitoring | Designed only, not built (see [Roadmap](#roadmap)) |

## What it predicts

```
target = log(1 + views) − log(1 + trailing_avg_views)
```

`trailing_avg_views` = mean views over the channel's 5 most recent prior uploads (shifted so a video never sees itself). Predicted views recovered via `invert_target`.

## Pipeline

```
YouTube Data API v3 ──► pipeline/run_ingestion.py ──► data/videos.csv + data/images/
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
   models/precompute_embeddings.py          models/precompute_visual_features.py
   (frozen CLIP/DINOv2/MiniLM)              (faces, OCR, color — dropped, see Ablations)
                 └───────────────────┬───────────────────┘
                                     ▼
                             models/train.py
                (late-fusion head, chronological split,
                 logs to experiments/results.jsonl)
```

## Data

YouTube Data API v3 (`playlistItems.list` + `videos.list`; `search.list` avoided — ~100x more quota). Hand-curated English-speaking personality/gaming/commentary channels (excludes news/release-driven channels), published 2025-01-01+, shorts (≤180s) excluded. Current: 10,591 usable rows (8,472 train / 1,059 val / 1,060 test).

**Retrieved:** thumbnail, title, views, duration, publish date, subscriber count, `categoryId` → genre.
**Derived:** title stats (length, caps, symbols, question mark, has-number), trailing average views, `is_first_video`, optional `clip_sim` (CLIP image/title cosine similarity).

## Model: v1 late fusion

- **Image:** CLIP ViT-B/32 (512-d), frozen. Alternatives: DINOv2 ViT-S/14, CLIP ViT-B/16.
- **Text:** CLIP text tower (512-d), frozen. Alternative: `all-MiniLM-L6-v2`.
- **Tabular:** log-scaled subscriber count + trailing views, duration, title stats, boolean flags, one-hot genre, optional `clip_sim`.
- **Fusion:** each branch → 32-d projection → concat → 64 → 16 → 1. Only projections + head train; embeddings precomputed once.
- **Training:** Huber loss, Adam (lr 2e-5, wd 1e-4), batch 64, dropout 0.2, embedding noise (std 0.02), early stopping (patience 10, max 200 epochs). Chronological 80/10/10 split by `published_at`; scalers fit on train only.

### Results

Mean ± std over seeds, same 1,060-row test split (`experiments/results.jsonl`):

| Text encoder | `clip_sim` | Test Spearman | Test AUC |
|---|---|---|---|
| CLIP | off | 0.335 ± 0.015 | 0.644 |
| CLIP | on | 0.333 ± 0.016 | 0.640 |
| MiniLM | off | 0.321 ± 0.012 | 0.641 |
| MiniLM | on | 0.327 ± 0.003 | 0.643 |

Differences between encoders/`clip_sim` are within seed noise.

### Ablations

**Modality** (single seed): tabular-only 0.277 Spearman → +image 0.302 → +text 0.308 → full fusion 0.317. Most skill comes from channel-level features; thumbnail+title add ~+0.04.

**Visual tabular features** (5 seeds, embeddings fixed): dropping face/OCR/color-stat columns showed no measurable benefit (Δ within noise, non-monotonic). **Decision: dropped all 7 from training and serving.**

## Known limitations

- `subscriber_count_at_upload` is the *current* count, not at-publish (TubeCensus integration shelved — Windows permission issues, storage cost). Fixed going forward by the Postgres redesign.
- Label timing isn't a fixed horizon yet — `label_finalized` flips at ≥28 days *at ingestion time*, not exactly day 28. Planned redesign fixes this.
- `compute_trailing_views` may compute a truncated history on incremental re-runs (found in review, not yet verified in practice) — full per-channel history in Postgres fixes this.
- `is_first_video` means first video in the fetched 2025+ window, not the channel's actual first upload.

## Serving

`export_bundle()` (in `train.py`) packages weights, scalers, feature-column order, and encoder config into one versioned file; `tests/test_bundle_parity.py` checks the serving path reproduces it (passing, max diff 2.8e-3 vs. a 1e-2 bug threshold). Serving is **CLIP-only** — a bundle trained with a different encoder is rejected at load time.

### Running it — Docker Compose (primary workflow)

This is how the app actually runs, both locally and on the eventual Oracle VM: FastAPI behind Caddy, on Caddy's internal Docker network.

```bash
docker compose up -d --build
docker compose ps          # api should show "healthy" after ~30s
```

The `api` service is intentionally **not** published to the host (`expose: ["8000"]`, not `ports:`) — only `caddy` is reachable from outside the Compose network, on 80/443. That's correct for production (nobody should be able to skip Caddy and hit the model directly), but it also means `curl http://localhost:8000/...` from the host will **not** connect, even when everything is healthy. To check the API directly during local dev, curl it from inside the Compose network instead:

```bash
docker compose exec caddy wget -qO- http://api:8000/health
```

`Caddyfile` is currently keyed to a real domain (`your-domain.example.com`), so Caddy itself won't respond to `localhost` until that's swapped for a real domain pointed at the VM. Since there's no domain to test against yet, Caddy itself can't be meaningfully exercised locally — only `api` can be.

**Testing `/predict` locally with a real curl command (temporary, revert before deploying):** the `wget` trick above only works for simple GETs like `/health`; `/predict` needs multipart file upload, which is easiest to test the normal way from the host. To allow that, temporarily change `api`'s `expose: ["8000"]` in `docker-compose.yml` to:
```yaml
    ports:
      - "8000:8000"
```
then `docker compose up -d` (no rebuild needed). This publishes the api port directly to the host, bypassing Caddy — convenient for local testing, but **it must be changed back to `expose: ["8000"]` before deploying to Oracle**. In production, nothing outside the Compose network should be able to reach `api` directly; only Caddy should be internet-facing. Don't ship this repo to the VM with `ports:` still set on `api`.

### Running it — plain Docker (quick local smoke test only, no Caddy)

Useful for a fast sanity check of the image itself without bringing up the whole stack — not the workflow this project actually deploys with.

```bash
# amd64 shown; swap --platform linux/arm64 for the Oracle VM
docker build -t ytpp-api .
docker run --rm -p 8000:8000 -v ./models/bundles:/app/models/bundles:ro ytpp-api

curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -F thumbnail=@some_image.jpg -F title="..." \
  -F subscriber_count_at_upload=482000 -F trailing_avg_views=310000 \
  -F duration_seconds=612 -F genre=Entertainment
```

Unlike the Compose workflow above, `-p 8000:8000` here does publish the port to the host directly, so plain `curl http://localhost:8000/...` works — but there's no Caddy/HTTPS in front of it, so this isn't representative of the deployed setup.

`requirements-serve.txt` is hand-curated and exactly pinned (not a `pip freeze`) to only what `serving/` imports. torch/torchvision install from PyTorch's CPU-only wheel index in the Dockerfile (a plain install resolves to CUDA wheels the Ampere VM, no GPU, doesn't need).

## Deployment (planned infra)

One Oracle Cloud Always Free Ampere A1 VM (arm64; 1 OCPU/6GB to start — resizable to 2/12 later without recreating), Docker Compose: Caddy (HTTPS) + FastAPI now, Postgres + Airflow + worker later. No AWS unless v2 training needs a GPU spot instance. Compose file kept portable (should run on EC2 if the free tier changes).

**Planned DAGs:** `ingest_new` (daily) → `embed_new` → `finalize_labels` (~28-day horizon) → `retrain` (periodic/on drift, hard gate + paired-bootstrap soft comparison, ties go to newer) → `monitor_drift` (weekly) → `backup` (nightly, off-VM).

## Roadmap

1. ~~Drop the 7 visual tabular columns~~ — done (see Ablations).
2. ~~Export versioned bundle + parity test~~ — done.
3. ~~Slim serving reqs, arm64 Dockerfile, validate Docker Compose stack locally~~ — done.
4. **Deploy to Oracle VM** — code and Compose stack validated locally (amd64); blocked on Oracle Ampere capacity, arm64 not yet tested on real hardware.
5. Move `videos.csv` → Postgres; containerize ingest/embed/finalize as CLI commands.
6. Wrap in Airflow DAGs; add `backup`.
7. Add `retrain` + promotion logic, then `monitor_drift`.
8. **v2, early fusion:** cross-attention transformer over image patches + title tokens (30-50k+ rows). Local GPU or AWS spot — undecided. Adopt only if measurably better than v1.

### Open decisions
- Retrain cadence, minimum new rows per cycle, test-slice length.
- Promotion rejection rule: strict CI-below-0, or with a margin.

## Repo structure

```
ingestion/     YouTube API client, thumbnail downloader, subscriber lookup (TubeCensus stub)
features/      Title features, trailing views, target, visual features
pipeline/      run_ingestion.py + config/ (channels.json, cum_channels.json)
models/        precompute_embeddings.py, dataset.py, late_fusion_model.py, train.py,
               baseline.py, ablation_*.py
serving/       bundle.py (LoadedBundle), features.py (feature reconstruction), app.py (FastAPI)
tests/         test_bundle_parity.py
experiments/   results.jsonl (one line per training run)
data/          local dataset output (gitignored)
Dockerfile, docker-compose.yml, Caddyfile, requirements-serve.txt, requirements-train.txt
```

## Setup

```bash
pip install -r requirements-train.txt   # full dev/training freeze (CUDA torch, Jupyter, TubeCensus deps)
```

`.env` (loaded via `python-dotenv`, before any project imports):
```
YOUTUBE_API_KEY=your_key_here
```

## Running it

```bash
python -m pipeline.run_ingestion         # 1. ingest (incremental, safe to re-run)
python models/late_fusion_model.py       # 2. sanity-check architecture
python -m models.precompute_embeddings   # 3. precompute embeddings
python -m models.train                   # 4. train (env vars: IMAGE_ENCODER, TEXT_ENCODER, USE_SIM, SEED)
python -m models.baseline                # 5. baselines / diagnostics
python -m models.ablation_modalities
```

Each run appends to `experiments/results.jsonl`; checkpoints saved to `models/checkpoints/`.