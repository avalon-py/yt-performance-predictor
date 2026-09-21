"""Rolling trailing-average-views feature -- must only look backward per channel."""


def compute_trailing_views(df, window=5):
    df = df.sort_values(["channel_id", "published_at"])
    df["trailing_avg_views"] = (
        df.groupby("channel_id")["views"]
        .transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
    )
    df["is_first_video"] = df["trailing_avg_views"].isna()
    return df