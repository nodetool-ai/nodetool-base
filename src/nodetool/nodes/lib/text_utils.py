from __future__ import annotations

"""
Utilities for incremental text streaming and overlap handling.
"""

from typing import List, Tuple


def _tokenize_whitespace(s: str) -> tuple[list[str], list[int]]:
    """Collapse consecutive whitespace into single-space tokens and map to original indices.

    Returns (tokens, end_index_map), where end_index_map[i] is the exclusive
    end-character index in the original string for tokens[i].
    """
    tokens: List[str] = []
    end_idx: List[int] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch.isspace():
            j = i + 1
            while j < n and s[j].isspace():
                j += 1
            tokens.append(" ")
            end_idx.append(j)
            i = j
        else:
            tokens.append(ch)
            end_idx.append(i + 1)
            i += 1
    return tokens, end_idx


def compute_incremental_suffix(
    previous_text: str,
    new_text: str,
    max_overlap: int = 4096,
    min_prefix_drop: int = 40,
) -> str:
    """
    Return the non-overlapping suffix of new_text when appended to previous_text.

    Improvements over a naive suffix/prefix match:
    - Whitespace-insensitive matching for robustness to spacing differences
    - Optional fallback: if the new prefix is repeated anywhere in previous_text,
      trim it when the repeated portion is long enough (min_prefix_drop)

    Args:
        previous_text: Accumulated text already emitted.
        new_text: The latest string from the decoder.
        max_overlap: Character window from the end of previous_text to consider.
        min_prefix_drop: Minimum characters to consider for global-prefix trimming.

    Returns:
        The incremental suffix to emit.
    """
    if not new_text:
        return ""

    # Limit window for efficiency
    window_prev = (
        previous_text[-max(0, max_overlap) :] if max_overlap > 0 else previous_text
    )

    # Fast path: if new_text already fully included in previous_text, nothing to emit
    if window_prev and new_text and new_text in previous_text:
        return ""

    # Whitespace-insensitive longest suffix/prefix match
    prev_tokens, _prev_map = _tokenize_whitespace(window_prev)
    new_tokens, new_map = _tokenize_whitespace(new_text)

    max_k = min(len(prev_tokens), len(new_tokens))
    matched_chars = 0
    for k in range(max_k, 0, -1):
        if prev_tokens[-k:] == new_tokens[:k]:
            # Map token-length match to original char index in new_text
            matched_chars = new_map[k - 1]
            break

    delta = new_text[matched_chars:]

    # Fallback: If we still have a long prefix already present anywhere in previous_text,
    # trim it once. This guards against minor punctuation/whitespace mismatches at the boundary.
    if delta and previous_text:
        # Consider up to max_overlap characters of the delta's prefix
        limit = min(len(delta), max_overlap if max_overlap > 0 else len(delta))
        # Try from longer to shorter, but not below the minimum threshold
        for k in range(limit, min_prefix_drop - 1, -1):
            prefix = delta[:k]
            if prefix in previous_text:
                delta = delta[k:]
                break

    return delta
