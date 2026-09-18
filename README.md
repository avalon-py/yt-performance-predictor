# YouTube Views Predictor

Predicting how well a YouTube video will perform — using only its thumbnail, title, and metadata available **before or at publish time**. No leakage from likes, comments, or post-publish signals.

## Idea

Given a video's thumbnail, title, and channel/tabular metadata, predict its relative view performance (vs. the channel's own baseline) at a fixed horizon after publishing.

## Approach

- **v1 — Late fusion**: separate pretrained encoders for the thumbnail (image) and title (text), concatenated into a small fusion head (MLP/regressor). Ships first — simpler, reuses transfer learning, gets the full pipeline working end-to-end.
- **v2 — Early fusion**: a single transformer with cross-attention between image patches and text tokens, trained on the accumulated dataset once there's enough volume. Warm-started from the same pretrained encoders where possible.

The project deliberately builds both, in sequence, and reports the comparison as the main technical result — not just "which model is better" but "when is the added complexity of cross-modal attention worth it."

## Features

**Image branch:** thumbnail pixels → CNN/ViT encoder

**Text branch:** title → text encoder. Description is reduced to engineered features (hashtag count, link count, length) rather than fed raw — weak/noisy signal.

**Tabular branch:**
- Channel subscriber count *at publish time*
- Channel's rolling average views over last N uploads (normalization baseline)
- Channel upload frequency (videos/week)
- Category/genre ID
- Duration + `is_short` flag
- Day-of-week and hour published
- Tag count
- Title length, word count, has-number, has-question-mark, all-caps ratio
- *(optional)* face detection on thumbnail: has-face, face count, text-overlay present

**Not used as features:** likes, comments — these accumulate on the same timeline as views, so using them as inputs is leakage.

## Target

```
log(views + 1) − log(channel_baseline + 1)
```
at a fixed horizon (e.g. 48h post-publish).

## Data Strategy

- **Backfill**: pull channels' full upload history via `playlistItems.list` (1 unit/call). Older videos have effectively "final" view counts — fast way to get tens of thousands of rows on day one.
- **Live stream**: an Airflow DAG captures new uploads at t=0 and snapshots the true label at exactly the target horizon — slower but exact.

Rough row targets: ~5,000–10,000 for v1 (late fusion), ~30,000–50,000+ for v2 (early fusion).

## Data Source

[YouTube Data API v3](https://developers.google.com/youtube/v3) — free tier, 10,000 units/day.

- `playlistItems.list` (1 unit) — new video IDs from a channel's uploads playlist
- `videos.list` (1 unit) — stats/metadata/thumbnails, up to 50 videos per call
- `search.list` (100 units) — avoided; too expensive for this use case

## Project Structure

```
.
├── ingestion/      # YouTube API client, backfill + live polling scripts
├── features/       # feature engineering (tabular, title, thumbnail preprocessing)
├── models/         # late-fusion (v1) and early-fusion (v2) model code
├── pipeline/        # Airflow DAG / orchestration for continuous data collection
├── data/           # raw + processed data (gitignored)
├── .env            # YOUTUBE_API_KEY (gitignored)
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/avalon-py/<repo-name>.git
cd <repo-name>
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Add your API key to `.env`:
```
YOUTUBE_API_KEY=your_key_here
```

## Status

🚧 Early stage — setting up ingestion pipeline.

## Roadmap

- [ ] YouTube API client with quota tracking
- [ ] Backfill script (historical upload data)
- [ ] Feature engineering pipeline
- [ ] v1 late-fusion model
- [ ] Live Airflow DAG for continuous labeling
- [ ] v2 early-fusion model
- [ ] v1 vs v2 comparison report