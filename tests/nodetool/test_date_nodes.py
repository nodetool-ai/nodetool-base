import pytest
from datetime import datetime, date, timezone, timedelta
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import Datetime, Date
from nodetool.nodes.lib.date import (
    Today,
    Now,
    ParseDate,
    ParseDateTime,
    AddTimeDelta,
    DateDifference,
    FormatDateTime,
    GetWeekday,
    DateRange,
    IsDateInRange,
    GetQuarter,
    DateToDatetime,
    DatetimeToDate,
    RelativeTime,
    TimeDirection,
    TimeUnitType,
    BoundaryTime,
    BoundaryType,
    PeriodType,
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


@pytest.mark.asyncio
async def test_parse_date(context: ProcessingContext):
    node = ParseDate(date_string="2024-05-15", input_format=DateFormat.ISO)
    result = await node.process(context)
    assert result.to_date() == date(2024, 5, 15)


@pytest.mark.asyncio
async def test_parse_datetime(context: ProcessingContext):
    node = ParseDateTime(
        datetime_string="2024-05-15T10:30:00", input_format=DateFormat.ISO_WITH_TIME
    )
    result = await node.process(context)
    # The result will have timezone info, so compare date and time components
    assert result.to_datetime().year == 2024
    assert result.to_datetime().month == 5
    assert result.to_datetime().day == 15
    assert result.to_datetime().hour == 10
    assert result.to_datetime().minute == 30


@pytest.mark.asyncio
async def test_add_time_delta(context: ProcessingContext):
    dt = Datetime.from_datetime(datetime(2024, 5, 15, 10, 0, 0))
    node = AddTimeDelta(input_datetime=dt, days=2, hours=3, minutes=15)
    result = await node.process(context)
    # Compare date and time components
    assert result.to_datetime().year == 2024
    assert result.to_datetime().month == 5
    assert result.to_datetime().day == 17
    assert result.to_datetime().hour == 13
    assert result.to_datetime().minute == 15


@pytest.mark.asyncio
async def test_date_difference(context: ProcessingContext):
    start = Datetime.from_datetime(datetime(2024, 5, 1, 8, 0, 0))
    end = Datetime.from_datetime(datetime(2024, 5, 3, 10, 30, 45))
    node = DateDifference(start_date=start, end_date=end)
    result = await node.process(context)
    assert result["days"] == 2
    assert result["hours"] == 2
    assert result["minutes"] == 30
    assert result["seconds"] == 45
    assert result["total_seconds"] > 0


@pytest.mark.asyncio
async def test_date_range(context: ProcessingContext):
    start = Datetime.from_datetime(datetime(2024, 5, 1))
    end = Datetime.from_datetime(datetime(2024, 5, 5))
    node = DateRange(start_date=start, end_date=end, step_days=1)
    result = await node.process(context)
    assert len(result) == 5
    assert result[0].to_datetime().date() == date(2024, 5, 1)
    assert result[4].to_datetime().date() == date(2024, 5, 5)


@pytest.mark.asyncio
async def test_date_range_with_step(context: ProcessingContext):
    start = Datetime.from_datetime(datetime(2024, 5, 1))
    end = Datetime.from_datetime(datetime(2024, 5, 10))
    node = DateRange(start_date=start, end_date=end, step_days=3)
    result = await node.process(context)
    assert len(result) == 4  # Days 1, 4, 7, 10
    assert result[0].to_datetime().date() == date(2024, 5, 1)
    assert result[3].to_datetime().date() == date(2024, 5, 10)


@pytest.mark.asyncio
async def test_get_quarter(context: ProcessingContext):
    # Test Q1
    dt = Datetime.from_datetime(datetime(2024, 2, 15))
    node = GetQuarter(input_datetime=dt)
    result = await node.process(context)
    assert result["quarter"] == 1
    
    # Test Q4
    dt = Datetime.from_datetime(datetime(2024, 12, 15))
    node = GetQuarter(input_datetime=dt)
    result = await node.process(context)
    assert result["quarter"] == 4


@pytest.mark.asyncio
async def test_date_to_datetime(context: ProcessingContext):
    input_date = Date.from_date(date(2024, 5, 15))
    node = DateToDatetime(input_date=input_date)
    result = await node.process(context)
    assert result.to_datetime().date() == date(2024, 5, 15)
    assert result.to_datetime().time() == datetime.min.time()


@pytest.mark.asyncio
async def test_datetime_to_date(context: ProcessingContext):
    dt = Datetime.from_datetime(datetime(2024, 5, 15, 14, 30, 0))
    node = DatetimeToDate(input_datetime=dt)
    result = await node.process(context)
    assert result.to_date() == date(2024, 5, 15)


@pytest.mark.asyncio
async def test_relative_time_future_days(context: ProcessingContext):
    node = RelativeTime(amount=3, unit=TimeUnitType.DAYS, direction=TimeDirection.FUTURE)
    result = await node.process(context)
    expected_date = (datetime.now() + timedelta(days=3)).date()
    assert result.to_datetime().date() == expected_date


@pytest.mark.asyncio
async def test_relative_time_past_hours(context: ProcessingContext):
    node = RelativeTime(amount=5, unit=TimeUnitType.HOURS, direction=TimeDirection.PAST)
    result = await node.process(context)
    expected = datetime.now(timezone.utc) - timedelta(hours=5)
    # Allow 2 seconds tolerance for test execution time
    assert abs((result.to_datetime() - expected).total_seconds()) < 2


@pytest.mark.asyncio
async def test_boundary_time(context: ProcessingContext):
    dt = Datetime.from_datetime(datetime(2024, 5, 15, 14, 30, 45))
    
    # Test start of day
    node = BoundaryTime(input_datetime=dt, period=PeriodType.DAY, boundary=BoundaryType.START)
    result = await node.process(context)
    assert result.to_datetime().hour == 0
    assert result.to_datetime().minute == 0
    assert result.to_datetime().second == 0
    
    # Test end of day
    node = BoundaryTime(input_datetime=dt, period=PeriodType.DAY, boundary=BoundaryType.END)
    result = await node.process(context)
    assert result.to_datetime().hour == 23
    assert result.to_datetime().minute == 59
    assert result.to_datetime().second == 59


@pytest.mark.asyncio
async def test_get_weekday_as_number(context: ProcessingContext):
    # Monday
    dt = Datetime.from_datetime(datetime(2023, 12, 25))
    node = GetWeekday(input_datetime=dt, as_name=False)
    result = await node.process(context)
    assert result == 0


@pytest.mark.asyncio
async def test_is_date_in_range_exclusive(context: ProcessingContext):
    check = Datetime.from_datetime(datetime(2024, 5, 1))
    start = Datetime.from_datetime(datetime(2024, 5, 1))
    end = Datetime.from_datetime(datetime(2024, 5, 10))
    node = IsDateInRange(check_date=check, start_date=start, end_date=end, inclusive=False)
    assert await node.process(context) is False
