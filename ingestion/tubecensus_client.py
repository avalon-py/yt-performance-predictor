"""
Subscriber count feature -- shelved TubeCensus for now (setup friction on
Windows + unconfirmed coverage for our 2025+ window wasn't worth chasing
before getting the rest of the pipeline running). Using current subscriber
count as an approximation.

Known limitation, tracked not hidden: for backfilled (older) videos, this
is the channel's CURRENT subscriber count, not their count at publish time.
Revisit later if it turns out to matter -- e.g. bucket into coarse tiers
(<100K / 100K-1M / 1M-5M / 5M-20M / 20M+) instead of an exact number, which
is far less sensitive to this staleness than the raw count is.
"""


def get_subscriber_count_at(channel_id, published_at, fallback_count):
    return fallback_count