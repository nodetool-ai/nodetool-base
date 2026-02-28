import pytest
from unittest.mock import patch
import datetime
from nodetool.nodes.nodetool.utils import generate_timestamped_name

class TestGenerateTimestampedName:
    @patch("nodetool.nodes.nodetool.utils.datetime")
    def test_basic_timestamp(self, mock_datetime):
        # Mock datetime.datetime.now()
        mock_now = datetime.datetime(2023, 10, 27, 10, 30, 45)
        mock_datetime.datetime.now.return_value = mock_now

        pattern = "file-%Y-%m-%d-%H-%M-%S.txt"
        expected = "file-2023-10-27-10-30-45.txt"

        result = generate_timestamped_name(pattern)
        assert result == expected

    @patch("nodetool.nodes.nodetool.utils.datetime")
    def test_no_format_codes(self, mock_datetime):
        mock_now = datetime.datetime(2023, 10, 27, 10, 30, 45)
        mock_datetime.datetime.now.return_value = mock_now

        pattern = "file.txt"
        expected = "file.txt"

        result = generate_timestamped_name(pattern)
        assert result == expected

    @patch("nodetool.nodes.nodetool.utils.datetime")
    def test_custom_format(self, mock_datetime):
        mock_now = datetime.datetime(2023, 1, 1, 0, 0, 0)
        mock_datetime.datetime.now.return_value = mock_now

        pattern = "report_%B_%Y.pdf"
        expected = "report_January_2023.pdf"

        result = generate_timestamped_name(pattern)
        assert result == expected
