import pytest
from nodetool.nodes.lib.text_utils import (
    compute_incremental_suffix,
    _tokenize_whitespace,
)


class TestTokenizeWhitespace:
    """Tests for _tokenize_whitespace function."""

    def test_empty_string(self):
        tokens, end_map = _tokenize_whitespace("")
        assert tokens == []
        assert end_map == []

    def test_single_character(self):
        tokens, end_map = _tokenize_whitespace("a")
        assert tokens == ["a"]
        assert end_map == [1]

    def test_multiple_characters(self):
        tokens, end_map = _tokenize_whitespace("abc")
        assert tokens == ["a", "b", "c"]
        assert end_map == [1, 2, 3]

    def test_single_space(self):
        tokens, end_map = _tokenize_whitespace(" ")
        assert tokens == [" "]
        assert end_map == [1]

    def test_multiple_spaces_collapsed(self):
        tokens, end_map = _tokenize_whitespace("a   b")
        assert tokens == ["a", " ", "b"]
        assert end_map == [1, 4, 5]

    def test_tabs_and_newlines_collapsed(self):
        tokens, end_map = _tokenize_whitespace("a\t\n\rb")
        assert tokens == ["a", " ", "b"]
        assert end_map == [1, 4, 5]

    def test_leading_trailing_whitespace(self):
        tokens, end_map = _tokenize_whitespace("  hello  ")
        assert tokens == [" ", "h", "e", "l", "l", "o", " "]
        assert end_map == [2, 3, 4, 5, 6, 7, 9]


class TestComputeIncrementalSuffix:
    """Tests for compute_incremental_suffix function."""

    def test_empty_new_text(self):
        result = compute_incremental_suffix("previous text", "")
        assert result == ""

    def test_empty_previous_text(self):
        result = compute_incremental_suffix("", "new text")
        assert result == "new text"

    def test_no_overlap(self):
        result = compute_incremental_suffix("hello", "world")
        assert result == "world"

    def test_full_overlap(self):
        result = compute_incremental_suffix("hello world", "hello world")
        assert result == ""

    def test_partial_overlap_suffix_prefix(self):
        result = compute_incremental_suffix("hello wor", "world!")
        # "wor" is the suffix of previous and prefix of new
        assert result == "ld!"

    def test_new_text_contained_in_previous(self):
        result = compute_incremental_suffix("the quick brown fox", "quick")
        assert result == ""

    def test_whitespace_insensitive_matching(self):
        # Previous ends with "hello ", new starts with "hello  " (extra space)
        result = compute_incremental_suffix("say hello ", "hello  world")
        # Should recognize "hello " matches "hello  " (whitespace collapsed)
        assert "world" in result

    def test_max_overlap_window(self):
        long_previous = "x" * 5000
        new_text = "y" * 100
        result = compute_incremental_suffix(long_previous, new_text, max_overlap=100)
        assert result == new_text

    def test_min_prefix_drop(self):
        # Test that prefix is dropped when found elsewhere in previous
        previous = "the prefix text appears at the end: prefix text"
        new_text = "prefix text more content"
        result = compute_incremental_suffix(previous, new_text, min_prefix_drop=5)
        # "prefix text" should be dropped since it appears in previous
        assert "more content" in result

    def test_incremental_streaming_pattern(self):
        # Simulate typical streaming pattern
        accumulated = "The weather today is"
        chunk1 = "today is sunny"
        result1 = compute_incremental_suffix(accumulated, chunk1)
        assert result1 == " sunny"

    def test_exact_suffix_match(self):
        previous = "Part A Part B"
        new_text = "Part B Part C"
        result = compute_incremental_suffix(previous, new_text)
        assert result == " Part C"

    def test_unicode_content(self):
        previous = "こんにちは"
        new_text = "にちは世界"
        result = compute_incremental_suffix(previous, new_text)
        assert "世界" in result

    def test_punctuation_handling(self):
        previous = "Hello, world"
        new_text = "world!"
        result = compute_incremental_suffix(previous, new_text)
        assert result == "!"
