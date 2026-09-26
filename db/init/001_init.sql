CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE videos (
    video_id                        TEXT PRIMARY KEY,
    channel_id                      TEXT NOT NULL,
    channel_ref                     TEXT NOT NULL,
    title                           TEXT NOT NULL,
    published_at                    TIMESTAMPTZ NOT NULL,
    duration_seconds                INTEGER NOT NULL,
    views                           BIGINT NOT NULL,
    label_finalized                 BOOLEAN NOT NULL,
    subscriber_count_at_upload      BIGINT,
    genre                           TEXT,
    thumbnail_path                  TEXT,
    title_length_chars              INTEGER,
    title_word_count                INTEGER,
    title_capitalized_word_count    INTEGER,
    title_capitalized_letter_count  INTEGER,
    title_capitalized_letter_ratio  DOUBLE PRECISION,
    title_symbol_count              INTEGER,
    title_has_question_mark         BOOLEAN,
    title_has_number                BOOLEAN,
    trailing_avg_views              DOUBLE PRECISION,
    is_first_video                  BOOLEAN,
    image_embedding                 VECTOR(512),
    text_embedding                  VECTOR(512)
);

CREATE INDEX idx_videos_channel_id ON videos (channel_id);