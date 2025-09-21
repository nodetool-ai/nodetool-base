from __future__ import annotations

"""
Utilities for incremental text streaming and overlap handling.
"""


def compute_incremental_suffix(
    previous_text: str,
    new_text: str,
    max_overlap: int = 1024,
) -> str:
    """
    Return the non-overlapping suffix of new_text when appended to previous_text.

    Many ASR/LLM streamers re-emit text with the same prefix as the existing
    transcript to maintain context. This function removes the longest prefix of
    new_text that is also a suffix of previous_text, so callers can emit only the
    incremental delta without repeating previously emitted content.

    Args:
        previous_text: The text that has already been emitted/accumulated.
        new_text: The latest text chunk returned by the model/decoder.
        max_overlap: Limit the suffix search window on previous_text for
            efficiency. Use 0 or a large value to consider the entire previous_text.

    Returns:
        The suffix of new_text that does not overlap with the end of previous_text.
        If new_text is entirely contained within previous_text, returns an empty string.
    """
    if not new_text:
        return ""

    if max_overlap > 0:
        window = previous_text[-max_overlap:]
    else:
        window = previous_text

    # Fast path: if new_text is already contained, nothing to emit
    if previous_text and new_text in previous_text:
        return ""

    # Find the longest k such that window.endswith(new_text[:k])
    max_k = min(len(new_text), len(window))
    for k in range(max_k, 0, -1):
        if window.endswith(new_text[:k]):
            return new_text[k:]

    # No overlap at boundary; emit full new_text
    return new_text
