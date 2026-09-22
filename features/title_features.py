"""Pure computations on title text and duration strings -- no API calls here."""

import re


def parse_duration_iso8601_to_seconds(duration):
    """Parse ISO 8601 duration (e.g. PT4M13S) into total seconds."""
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
    if not match:
        return 0
    h, m, s = (int(x) if x else 0 for x in match.groups())
    return h * 3600 + m * 60 + s


def title_features(title):
    words = title.split()
    letters = [c for c in title if c.isalpha()]

    return {
        "title_length_chars": len(title),
        "title_word_count": len(words),
        "title_capitalized_word_count": sum(1 for w in words if w.isupper() and len(w) > 1),
        "title_capitalized_letter_count": sum(1 for c in title if c.isupper()),
        "title_capitalized_letter_ratio": (
            sum(1 for c in title if c.isupper()) / len(letters) if letters else 0.0
        ),
        "title_symbol_count": len(re.findall(r"[!?$%#@]", title)),
        "title_has_question_mark": "?" in title,
        "title_has_number": bool(re.search(r"\d", title)),
    }