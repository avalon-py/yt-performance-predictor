"""
Loads precomputed embeddings (from precompute_embeddings.py) joined with
tabular features from the main CSV. No image/text encoder inference happens
here -- that's the whole point of caching, this is just fast tensor assembly.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

TABULAR_LOG_COLS = ["subscriber_count_at_upload", "trailing_avg_views"]
TABULAR_NUMERIC_COLS = [
    "duration_seconds", "title_length_chars", "title_word_count",
    "title_capitalized_word_count", "title_capitalized_letter_count",
    "title_capitalized_letter_ratio", "title_symbol_count",
    "face_count", "mean_saturation", "mean_brightness",
    "brightness_std", "warm_hue_ratio",
]
TABULAR_BOOL_COLS = [
    "title_has_question_mark", "title_has_number",
    "has_face", "has_text_overlay",
]


def build_tabular_matrix(df, genre_categories, scaler=None, fit_scaler=False):
    """Assemble the full tabular feature matrix. Fit the scaler only on train data,
    then reuse it (via fit_scaler=False) for val/test to avoid leaking their
    distribution into the scaling."""
    from sklearn.preprocessing import StandardScaler

    log_cols = df[TABULAR_LOG_COLS].astype(float).apply(np.log1p).values
    numeric_cols = df[TABULAR_NUMERIC_COLS].astype(float).values
    numeric_all = np.concatenate([log_cols, numeric_cols], axis=1)

    if fit_scaler:
        scaler = StandardScaler().fit(numeric_all)
    numeric_scaled = scaler.transform(numeric_all)

    bool_cols = df[TABULAR_BOOL_COLS].astype(float).values

    genre_onehot = np.zeros((len(df), len(genre_categories)), dtype=np.float32)
    for i, genre in enumerate(df["genre"].values):
        if genre in genre_categories:
            genre_onehot[i, genre_categories.index(genre)] = 1.0

    tabular = np.concatenate([numeric_scaled, bool_cols, genre_onehot], axis=1).astype(np.float32)
    return tabular, scaler


class VideoDataset(Dataset):
    def __init__(self, image_embeddings, text_embeddings, tabular, targets, video_ids):
        assert len(image_embeddings) == len(text_embeddings) == len(tabular) == len(targets)
        self.image_embeddings = image_embeddings
        self.text_embeddings = text_embeddings
        self.tabular = tabular
        self.targets = targets
        self.video_ids = video_ids

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "image_embedding": torch.tensor(self.image_embeddings[idx], dtype=torch.float32),
            "text_embedding": torch.tensor(self.text_embeddings[idx], dtype=torch.float32),
            "tabular": torch.tensor(self.tabular[idx], dtype=torch.float32),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
            "video_id": self.video_ids[idx],
        }