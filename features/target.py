"""
The regression target: log-relative performance vs. the channel's own recent
baseline, not raw views. This is the normalization decision from earlier --
raw views are dominated by channel size, this isolates "did this video over-
or under-perform its own channel's norm."
"""

import numpy as np


def compute_target(views, trailing_avg_views):
    return np.log1p(views) - np.log1p(trailing_avg_views)


def invert_target(target_value, trailing_avg_views):
    """Convert a predicted target back into an estimated view count."""
    return np.expm1(target_value + np.log1p(trailing_avg_views))