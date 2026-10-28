"""Keep internal evidence identifiers out of user-facing answer text."""

import re
from collections.abc import Iterable


def strip_chunk_ids(answer: str, known_ids: Iterable[str]) -> str:
    """Remove citation markers, leaving structured citation metadata untouched.

    Match exact IDs from this response as well as the project's generated ID
    format, so fabricated references cannot leak either. Do not remove arbitrary
    bracketed prose, numeric references, or ordinary Markdown links.
    """
    ids = sorted({value for value in known_ids if value}, key=len, reverse=True)
    alternatives = [re.escape(value) for value in ids]
    alternatives.append(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*_[a-f0-9]{8}_[0-9]{2,}")
    identifier = rf"(?<![a-zA-Z0-9_-])(?:{'|'.join(alternatives)})(?![a-zA-Z0-9_-])"
    # A readable link label can survive after its internal-only target is removed.
    answer = re.sub(rf"\[([^\]\n]*)\]\(\s*#?{identifier}\s*\)", r"\1", answer)
    group = rf"{identifier}(?:\s*[,，、;；]\s*{identifier})*"
    markers = (
        rf"\[\s*{group}\s*\](?:\([^\n)]*\))?",
        rf"【\s*{group}\s*】",
        rf"`{group}`",
        rf"\(\s*{group}\s*\)",
        identifier,
    )
    return re.sub(rf"[ \t]*(?:{'|'.join(markers)})", "", answer).strip()
