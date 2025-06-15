import pytest
from datetime import datetime, date, timezone
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import Datetime
from nodetool.nodes.lib.date import (
    Today,
    Now,
    FormatDateTime,
    GetWeekday,
    IsDateInRange,
    DateFormat,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_today_and_now(context: ProcessingContext):
    before = datetime.now(timezone.utc)
    today_node = Today()
    now_node = Now()
    today = await today_node.process(context)
    now = await now_node.process(context)
    after = datetime.now(timezone.utc)
    assert today.to_date() == date.today()
    assert before <= now.to_datetime() <= after


@pytest.mark.asyncio
async def test_format_and_weekday(context: ProcessingContext):
    dt = Datetime.from_datetime(datetime(2023, 12, 25, 8, 30, 0))
    format_node = FormatDateTime(input_datetime=dt, output_format=DateFormat.ISO)
    weekday_node = GetWeekday(input_datetime=dt, as_name=True)
    formatted = await format_node.process(context)
    weekday = await weekday_node.process(context)
    assert formatted == "2023-12-25"
    assert weekday == "Monday"


@pytest.mark.asyncio
async def test_is_date_in_range(context: ProcessingContext):
    check = Datetime.from_datetime(datetime(2024, 5, 5))
    start = Datetime.from_datetime(datetime(2024, 5, 1))
    end = Datetime.from_datetime(datetime(2024, 5, 10))
    node = IsDateInRange(check_date=check, start_date=start, end_date=end)
    assert await node.process(context) is True
