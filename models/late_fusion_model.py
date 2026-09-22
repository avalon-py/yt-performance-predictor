"""
Late fusion: each modality gets its own frozen encoder (embeddings precomputed
separately, see precompute_embeddings.py), a small learned projection per
branch, then everything is concatenated before a final regression head.

Only the projections + fusion head are trained in v1 -- the encoders that
produced image_embedding/text_embedding are frozen and never touched here.
"""

import torch
import torch.nn as nn


class LateFusionModel(nn.Module):
    def __init__(self, image_dim, text_dim, tabular_dim,
                 proj_dim=256, tabular_proj_dim=64, dropout=0.3):
        super().__init__()

        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.tabular_proj = nn.Sequential(
            nn.Linear(tabular_dim, tabular_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        fusion_input_dim = proj_dim * 2 + tabular_proj_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, image_embedding, text_embedding, tabular):
        img = self.image_proj(image_embedding)
        txt = self.text_proj(text_embedding)
        tab = self.tabular_proj(tabular)
        fused = torch.cat([img, txt, tab], dim=1)
        return self.fusion_head(fused).squeeze(-1)


if __name__ == "__main__":
    # Quick shape sanity check with synthetic tensors -- no pretrained weights needed.
    batch_size = 8
    image_dim, text_dim, tabular_dim = 768, 384, 15

    model = LateFusionModel(image_dim, text_dim, tabular_dim)
    dummy_image = torch.randn(batch_size, image_dim)
    dummy_text = torch.randn(batch_size, text_dim)
    dummy_tabular = torch.randn(batch_size, tabular_dim)

    output = model(dummy_image, dummy_text, dummy_tabular)
    assert output.shape == (batch_size,), f"unexpected output shape: {output.shape}"
    print(f"OK -- output shape {output.shape}, sample values: {output[:3].tolist()}")