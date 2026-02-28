import sys
import unittest
from unittest.mock import MagicMock
from datetime import datetime, date, timedelta, timezone
import os
from pydantic import BaseModel, ConfigDict

# Add src to sys.path so we can import the node
# Current file: /app/tests/nodetool/test_boundary_time_standalone.py
# Target: /app/src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

# Mock sys.modules for nodetool.workflows.base_node
mock_base_node = MagicMock()
class MockBaseNode(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _expose_as_tool: bool = True
mock_base_node.BaseNode = MockBaseNode
sys.modules["nodetool.workflows.base_node"] = mock_base_node

# Mock nodetool.workflows.processing_context
mock_processing_context = MagicMock()
class ProcessingContext:
    def __init__(self, user_id=None, auth_token=None):
        self.user_id = user_id
        self.auth_token = auth_token
mock_processing_context.ProcessingContext = ProcessingContext
sys.modules["nodetool.workflows.processing_context"] = mock_processing_context

# Mock nodetool.metadata.types
mock_metadata_types = MagicMock()
class MockDatetime:
    def __init__(self, dt=None):
        self._dt = dt or datetime.now(timezone.utc)

    def to_datetime(self):
        return self._dt

    @classmethod
    def from_datetime(cls, dt):
        return cls(dt)

    def __eq__(self, other):
        if isinstance(other, MockDatetime):
            return self._dt == other._dt
        if isinstance(other, datetime):
            return self._dt == other
        return False

    def __repr__(self):
        return f"Datetime({self._dt})"

class MockDate:
    def __init__(self, d=None):
        self._d = d or date.today()

    def to_date(self):
        return self._d

    @classmethod
    def from_date(cls, d):
        return cls(d)

mock_metadata_types.Datetime = MockDatetime
mock_metadata_types.Date = MockDate
sys.modules["nodetool.metadata.types"] = mock_metadata_types

# Now import the node
from nodetool.nodes.lib.date import BoundaryTime, PeriodType, BoundaryType

class TestBoundaryTime(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.context = ProcessingContext()

    async def test_day_boundary(self):
        # 2023-10-25 is a Wednesday
        dt = datetime(2023, 10, 25, 14, 30, 45, 123456)
        input_dt = MockDatetime(dt)

        # Start of day
        node = BoundaryTime(input_datetime=input_dt, period=PeriodType.DAY, boundary=BoundaryType.START)
        result = await node.process(self.context)
        self.assertEqual(result.to_datetime(), datetime(2023, 10, 25, 0, 0, 0))

        # End of day
        node = BoundaryTime(input_datetime=input_dt, period=PeriodType.DAY, boundary=BoundaryType.END)
        result = await node.process(self.context)
        self.assertEqual(result.to_datetime(), datetime(2023, 10, 25, 23, 59, 59, 999999))

    async def test_week_boundary_monday_start(self):
        # 2023-10-25 is Wednesday.
        # Week starts Monday 2023-10-23. Ends Sunday 2023-10-29.
        dt = datetime(2023, 10, 25, 12, 0, 0)
        input_dt = MockDatetime(dt)

        # Start of week (Monday)
        node = BoundaryTime(
            input_datetime=input_dt,
            period=PeriodType.WEEK,
            boundary=BoundaryType.START,
            start_monday=True
        )
        result = await node.process(self.context)
        self.assertEqual(result.to_datetime(), datetime(2023, 10, 23, 0, 0, 0))

        # End of week (Sunday)
        node = BoundaryTime(
            input_datetime=input_dt,
            period=PeriodType.WEEK,
            boundary=BoundaryType.END,
            start_monday=True
        )
        result = await node.process(self.context)
        self.assertEqual(result.to_datetime(), datetime(2023, 10, 29, 23, 59, 59, 999999))

        # Test edge case: Input is Monday
        dt_monday = datetime(2023, 10, 23, 10, 0, 0)
        node = BoundaryTime(input_datetime=MockDatetime(dt_monday), period=PeriodType.WEEK, boundary=BoundaryType.START, start_monday=True)
        result = await node.process(self.context)
        self.assertEqual(result.to_datetime(), datetime(2023, 10, 23, 0, 0, 0))

        # Test edge case: Input is Sunday
        dt_sunday = datetime(2023, 10, 29, 10, 0, 0)
        node = BoundaryTime(input_datetime=MockDatetime(dt_sunday), period=PeriodType.WEEK, boundary=BoundaryType.END, start_monday=True)
        result = await node.process(self.context)
        self.assertEqual(result.to_datetime(), datetime(2023, 10, 29, 23, 59, 59, 999999))

    async def test_week_boundary_sunday_start(self):
        # 2023-10-25 is Wednesday.
        # Week starts Sunday 2023-10-22. Ends Saturday 2023-10-28.
        dt = datetime(2023, 10, 25, 12, 0, 0)
        input_dt = MockDatetime(dt)

        # Start of week (Sunday)
        node = BoundaryTime(
            input_datetime=input_dt,
            period=PeriodType.WEEK,
            boundary=BoundaryType.START,
            start_monday=False
        )
        result = await node.process(self.context)
        self.assertEqual(result.to_datetime(), datetime(2023, 10, 22, 0, 0, 0))

        # End of week (Saturday)
        node = BoundaryTime(
            input_datetime=input_dt,
            period=PeriodType.WEEK,
            boundary=BoundaryType.END,
            start_monday=False
        )
        result = await node.process(self.context)
        self.assertEqual(result.to_datetime(), datetime(2023, 10, 28, 23, 59, 59, 999999))

        # Test edge case: Input is Sunday
        dt_sunday = datetime(2023, 10, 22, 10, 0, 0)
        node = BoundaryTime(input_datetime=MockDatetime(dt_sunday), period=PeriodType.WEEK, boundary=BoundaryType.START, start_monday=False)
        result = await node.process(self.context)
        self.assertEqual(result.to_datetime(), datetime(2023, 10, 22, 0, 0, 0))

        # Test edge case: Input is Saturday
        dt_saturday = datetime(2023, 10, 28, 10, 0, 0)
        node = BoundaryTime(input_datetime=MockDatetime(dt_saturday), period=PeriodType.WEEK, boundary=BoundaryType.END, start_monday=False)
        result = await node.process(self.context)
        self.assertEqual(result.to_datetime(), datetime(2023, 10, 28, 23, 59, 59, 999999))

    async def test_month_boundary(self):
        # 31-day month: October
        dt = datetime(2023, 10, 15, 12, 0, 0)
        input_dt = MockDatetime(dt)

        node = BoundaryTime(input_datetime=input_dt, period=PeriodType.MONTH, boundary=BoundaryType.START)
        result = await node.process(self.context)
        self.assertEqual(result.to_datetime(), datetime(2023, 10, 1, 0, 0, 0))

        node = BoundaryTime(input_datetime=input_dt, period=PeriodType.MONTH, boundary=BoundaryType.END)
        result = await node.process(self.context)
        self.assertEqual(result.to_datetime(), datetime(2023, 10, 31, 23, 59, 59, 999999))

        # 30-day month: April
        dt_apr = datetime(2023, 4, 15, 12, 0, 0)
        node = BoundaryTime(input_datetime=MockDatetime(dt_apr), period=PeriodType.MONTH, boundary=BoundaryType.END)
        result = await node.process(self.context)
        self.assertEqual(result.to_datetime(), datetime(2023, 4, 30, 23, 59, 59, 999999))

        # February (non-leap year)
        dt_feb = datetime(2023, 2, 15, 12, 0, 0)
        node = BoundaryTime(input_datetime=MockDatetime(dt_feb), period=PeriodType.MONTH, boundary=BoundaryType.END)
        result = await node.process(self.context)
        self.assertEqual(result.to_datetime(), datetime(2023, 2, 28, 23, 59, 59, 999999))

        # February (leap year)
        dt_feb_leap = datetime(2024, 2, 15, 12, 0, 0)
        node = BoundaryTime(input_datetime=MockDatetime(dt_feb_leap), period=PeriodType.MONTH, boundary=BoundaryType.END)
        result = await node.process(self.context)
        self.assertEqual(result.to_datetime(), datetime(2024, 2, 29, 23, 59, 59, 999999))

        # December (year boundary check)
        dt_dec = datetime(2023, 12, 15, 12, 0, 0)
        node = BoundaryTime(input_datetime=MockDatetime(dt_dec), period=PeriodType.MONTH, boundary=BoundaryType.END)
        result = await node.process(self.context)
        self.assertEqual(result.to_datetime(), datetime(2023, 12, 31, 23, 59, 59, 999999))

    async def test_year_boundary(self):
        dt = datetime(2023, 6, 15, 12, 0, 0)
        input_dt = MockDatetime(dt)

        # Start of year
        node = BoundaryTime(input_datetime=input_dt, period=PeriodType.YEAR, boundary=BoundaryType.START)
        result = await node.process(self.context)
        self.assertEqual(result.to_datetime(), datetime(2023, 1, 1, 0, 0, 0))

        # End of year
        node = BoundaryTime(input_datetime=input_dt, period=PeriodType.YEAR, boundary=BoundaryType.END)
        result = await node.process(self.context)
        self.assertEqual(result.to_datetime(), datetime(2023, 12, 31, 23, 59, 59, 999999))

if __name__ == '__main__':
    unittest.main()
