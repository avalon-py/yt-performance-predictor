"""
Subscriber-count-at-publish-time lookup via TubeCensus.

TODO: this is a sketch, not verified code. Confirm the actual package name
and function signature against TubeCensus's own docs/README before relying
on this in the pipeline. Currently falls back to current subscriber count
if the lookup isn't available, which reintroduces the staleness issue for
backfilled videos -- don't ship this fallback silently into production data
without knowing when it's firing.
"""


def get_subscriber_count_at(channel_id, published_at, fallback_count):
    try:
        import tubecensus  # confirm actual import name
        # snapshot = tubecensus.lookup(channel_id, date=published_at)
        # return snapshot.subscriber_count
        raise NotImplementedError
    except Exception:
        return fallback_count