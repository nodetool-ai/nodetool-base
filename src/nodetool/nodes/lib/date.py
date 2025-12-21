from datetime import datetime, timedelta, date
from enum import Enum
from pydantic import Field
from typing import ClassVar, TypedDict
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import Datetime, Date


class DateFormat(str, Enum):
    ISO = "%Y-%m-%d"
    US = "%m/%d/%Y"
    EUROPEAN = "%d/%m/%Y"
    HUMAN_READABLE = "%B %d, %Y"
    COMPACT = "%Y%m%d"
    FILENAME = "%Y%m%d_%H%M%S"
    ISO_WITH_TIME = "%Y-%m-%dT%H:%M:%S"
    ISO_WITH_TIMEZONE = "%Y-%m-%dT%H:%M:%S%z"
    ISO_WITH_TIMEZONE_UTC = "%Y-%m-%dT%H:%M:%S%z"


class Today(BaseNode):
    """
    Get current date without time component.

    Returns today's date in UTC timezone with time set to midnight.

    Returns: Date object representing today

    Typical usage: Generate timestamps, create date-based filenames, or filter by
    current date. Follow with date formatting, date arithmetic, or comparison nodes.

    date, today, now
    """

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Date:
        return Date.from_date(date.today())


class Now(BaseNode):
    """
    Get current date and time with full precision.

    Returns the current moment in UTC timezone with date and time components.

    Returns: Datetime object representing the current instant

    Typical usage: Generate precise timestamps, measure durations, or create
    time-based records. Follow with datetime formatting, time arithmetic, or
    comparison nodes.

    datetime, current, now
    """

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> Datetime:
        from datetime import timezone

        return Datetime.from_datetime(datetime.now(timezone.utc))


class ParseDate(BaseNode):
    """
    Parse date string into Date object using specified format.

    Converts a date string in a known format to a structured Date object.
    Supports multiple common date formats (ISO, US, European, etc.).

    Parameters:
    - date_string (required): Date string to parse
    - input_format (optional, default=ISO): Format pattern matching the input string

    Returns: Date object

    Raises: ValueError if string doesn't match format

    Typical usage: Convert date strings from APIs, parse user input, or normalize
    date formats. Follow with date arithmetic, formatting, or comparison nodes.

    date, parse, format
    """

    _expose_as_tool: ClassVar[bool] = True

    date_string: str = Field(default="", description="The date string to parse")
    input_format: DateFormat = Field(
        default=DateFormat.ISO, description="Format of the input date string"
    )

    async def process(self, context: ProcessingContext) -> Date:
        return Date.from_date(
            datetime.strptime(self.date_string, self.input_format.value)
        )


class ParseDateTime(BaseNode):
    """
    Parse datetime string into Datetime object using specified format.

    Converts a datetime string in a known format to a structured Datetime object
    with both date and time components. Supports multiple common datetime formats.

    Parameters:
    - datetime_string (required): Datetime string to parse
    - input_format (optional, default=ISO): Format pattern matching the input string

    Returns: Datetime object

    Raises: ValueError if string doesn't match format

    Typical usage: Convert datetime strings from APIs, parse timestamps, or normalize
    datetime formats. Follow with time arithmetic, formatting, or comparison nodes.

    datetime, parse, format
    """

    _expose_as_tool: ClassVar[bool] = True

    datetime_string: str = Field(default="", description="The datetime string to parse")
    input_format: DateFormat = Field(
        default=DateFormat.ISO, description="Format of the input datetime string"
    )

    async def process(self, context: ProcessingContext) -> Datetime:
        return Datetime.from_datetime(
            datetime.strptime(self.datetime_string, self.input_format.value)
        )


class AddTimeDelta(BaseNode):
    """
    Add or subtract time offset from a datetime.

    Adjusts a datetime by adding or subtracting days, hours, and minutes.
    Use negative values to subtract time.

    Parameters:
    - input_datetime (required): Starting datetime to modify
    - days (optional, default=0): Days to add (-3650 to 3650)
    - hours (optional, default=0): Hours to add (-24 to 24)
    - minutes (optional, default=0): Minutes to add (-60 to 60)

    Returns: New Datetime with offset applied

    Typical usage: Calculate future or past dates, schedule events, generate date
    ranges, or implement relative time logic. Follow with datetime formatting or
    comparison nodes.

    datetime, add, subtract
    """

    _expose_as_tool: ClassVar[bool] = True

    input_datetime: Datetime = Field(
        default=Datetime(), description="Starting datetime"
    )
    days: int = Field(
        ge=-365 * 10,
        le=365 * 10,
        default=0,
        description="Number of days to add (negative to subtract)",
    )
    hours: int = Field(
        ge=-24,
        le=24,
        default=0,
        description="Number of hours to add (negative to subtract)",
    )
    minutes: int = Field(
        ge=-60,
        le=60,
        default=0,
        description="Number of minutes to add (negative to subtract)",
    )

    async def process(self, context: ProcessingContext) -> Datetime:
        delta = timedelta(days=self.days, hours=self.hours, minutes=self.minutes)
        return Datetime.from_datetime(self.input_datetime.to_datetime() + delta)


class DateDifference(BaseNode):
    """
    Calculate time duration between two datetimes.

    Computes the difference between start and end datetimes, returning the
    interval broken down into multiple time units.

    Parameters:
    - start_date (required): Beginning datetime
    - end_date (required): Ending datetime

    Returns: Dictionary with "total_seconds" (int), "days" (int), "hours" (int),
    "minutes" (int), and "seconds" (int) representing the duration

    Typical usage: Measure elapsed time, calculate event durations, or determine
    time intervals. Follow with arithmetic, comparison, or conditional logic nodes.

    datetime, difference, duration
    """

    _expose_as_tool: ClassVar[bool] = True

    start_date: Datetime = Field(default=Datetime(), description="Start datetime")
    end_date: Datetime = Field(default=Datetime(), description="End datetime")

    class OutputType(TypedDict):
        total_seconds: int
        days: int
        hours: int
        minutes: int
        seconds: int

    async def process(self, context: ProcessingContext) -> OutputType:
        diff = self.end_date.to_datetime() - self.start_date.to_datetime()
        return {
            "total_seconds": int(diff.total_seconds()),
            "days": diff.days,
            "hours": diff.seconds // 3600,
            "minutes": (diff.seconds % 3600) // 60,
            "seconds": diff.seconds % 60,
        }


class FormatDateTime(BaseNode):
    """
    Convert a datetime object to a formatted string.
    datetime, format, convert

    Use cases:
    - Standardize date formats
    - Prepare dates for different systems
    """

    _expose_as_tool: ClassVar[bool] = True

    input_datetime: Datetime = Field(
        default=Datetime(),
        description="Datetime object to format",
    )
    output_format: DateFormat = Field(
        default=DateFormat.HUMAN_READABLE, description="Desired output format"
    )

    async def process(self, context: ProcessingContext) -> str:
        return self.input_datetime.to_datetime().strftime(self.output_format.value)


class GetWeekday(BaseNode):
    """
    Get the weekday name or number from a datetime.
    datetime, weekday, name

    Use cases:
    - Get day names for scheduling
    - Filter events by weekday
    """

    _expose_as_tool: ClassVar[bool] = True

    input_datetime: Datetime = Field(default=Datetime(), description="Input datetime")
    as_name: bool = Field(
        default=True, description="Return weekday name instead of number (0-6)"
    )

    async def process(self, context: ProcessingContext) -> str | int:
        if self.as_name:
            return self.input_datetime.to_datetime().strftime("%A")
        return self.input_datetime.to_datetime().weekday()


class DateRange(BaseNode):
    """
    Generate a list of dates between start and end dates.
    datetime, range, list

    Use cases:
    - Generate date sequences
    - Create date-based iterations
    """

    _expose_as_tool: ClassVar[bool] = True

    start_date: Datetime = Field(
        default=Datetime(),
        description="Start date of the range",
    )
    end_date: Datetime = Field(
        default=Datetime(),
        description="End date of the range",
    )
    step_days: int = Field(default=1, description="Number of days between each date")

    async def process(self, context: ProcessingContext) -> list[Datetime]:
        dates = []
        current = self.start_date
        while current.to_datetime() <= self.end_date.to_datetime():
            dates.append(current)
            current = Datetime.from_datetime(
                current.to_datetime() + timedelta(days=self.step_days)
            )
        return dates


class IsDateInRange(BaseNode):
    """
    Check if a date falls within a specified range.
    datetime, range, check

    Use cases:
    - Validate date ranges
    - Filter date-based data
    """

    _expose_as_tool: ClassVar[bool] = True

    check_date: Datetime = Field(default=Datetime(), description="Date to check")
    start_date: Datetime = Field(
        default=Datetime(),
        description="Start of date range",
    )
    end_date: Datetime = Field(
        default=Datetime(),
        description="End of date range",
    )
    inclusive: bool = Field(
        default=True, description="Include start and end dates in range"
    )

    async def process(self, context: ProcessingContext) -> bool:
        if self.inclusive:
            return (
                self.start_date.to_datetime()
                <= self.check_date.to_datetime()
                <= self.end_date.to_datetime()
            )
        return (
            self.start_date.to_datetime()
            < self.check_date.to_datetime()
            < self.end_date.to_datetime()
        )


class GetQuarter(BaseNode):
    """
    Get the quarter number and start/end dates for a given datetime.
    datetime, quarter, period

    Use cases:
    - Financial reporting periods
    - Quarterly analytics
    """

    _expose_as_tool: ClassVar[bool] = True

    input_datetime: Datetime = Field(default=Datetime(), description="Input datetime")

    class OutputType(TypedDict):
        quarter: int
        quarter_start: Datetime
        quarter_end: Datetime

    async def process(self, context: ProcessingContext) -> OutputType:
        quarter = (self.input_datetime.month - 1) // 3 + 1
        quarter_start = datetime(self.input_datetime.year, 3 * quarter - 2, 1)

        if quarter == 4:
            quarter_end = Datetime.from_datetime(
                datetime(self.input_datetime.year + 1, 1, 1) - timedelta(days=1)
            )
        else:
            quarter_end = Datetime.from_datetime(
                datetime(self.input_datetime.year, 3 * quarter + 1, 1)
                - timedelta(days=1)
            )

        return {
            "quarter": quarter,
            "quarter_start": Datetime.from_datetime(quarter_start),
            "quarter_end": quarter_end,
        }


class DateToDatetime(BaseNode):
    """
    Convert a Date object to a Datetime object.
    date, datetime, convert
    """

    _expose_as_tool: ClassVar[bool] = True

    input_date: Date = Field(default=Date(), description="Date to convert")

    async def process(self, context: ProcessingContext) -> Datetime:
        return Datetime.from_datetime(
            datetime.combine(self.input_date.to_date(), datetime.min.time())
        )


class DatetimeToDate(BaseNode):
    """
    Convert a Datetime object to a Date object.
    date, datetime, convert
    """

    _expose_as_tool: ClassVar[bool] = True

    input_datetime: Datetime = Field(
        default=Datetime(),
        description="Datetime to convert",
    )

    async def process(self, context: ProcessingContext) -> Date:
        return Date.from_date(self.input_datetime.to_datetime().date())


class TimeDirection(str, Enum):
    PAST = "past"
    FUTURE = "future"


class TimeUnitType(str, Enum):
    HOURS = "hours"
    DAYS = "days"
    MONTHS = "months"


class RelativeTime(BaseNode):
    """
    Get datetime relative to current time (past or future).
    datetime, past, future, relative, hours, days, months

    Use cases:
    - Calculate past or future dates
    - Generate relative timestamps
    """

    _expose_as_tool: ClassVar[bool] = True

    amount: int = Field(ge=0, default=1, description="Amount of time units")
    unit: TimeUnitType = Field(default=TimeUnitType.DAYS, description="Time unit type")
    direction: TimeDirection = Field(
        default=TimeDirection.FUTURE, description="Past or future"
    )

    async def process(self, context: ProcessingContext) -> Datetime:
        current = datetime.now()

        if self.unit == TimeUnitType.HOURS:
            delta = timedelta(hours=self.amount)
            if self.direction == TimeDirection.PAST:
                return Datetime.from_datetime(current - delta)
            else:
                return Datetime.from_datetime(current + delta)

        elif self.unit == TimeUnitType.DAYS:
            delta = timedelta(days=self.amount)
            if self.direction == TimeDirection.PAST:
                return Datetime.from_datetime(current - delta)
            else:
                return Datetime.from_datetime(current + delta)

        else:  # TimeUnitType.MONTHS
            year = current.year
            month = current.month

            if self.direction == TimeDirection.PAST:
                month -= self.amount
                while month <= 0:
                    month += 12
                    year -= 1
            else:
                month += self.amount
                while month > 12:
                    month -= 12
                    year += 1

            return Datetime.from_datetime(current.replace(year=year, month=month))


class BoundaryType(str, Enum):
    START = "start"
    END = "end"


class PeriodType(str, Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class BoundaryTime(BaseNode):
    """
    Get the start or end of a time period (day, week, month, year).
    datetime, start, end, boundary, day, week, month, year

    Use cases:
    - Get period boundaries for reporting
    - Normalize dates to period starts/ends
    """

    _expose_as_tool: ClassVar[bool] = True

    input_datetime: Datetime = Field(default=Datetime(), description="Input datetime")
    period: PeriodType = Field(default=PeriodType.DAY, description="Time period type")
    boundary: BoundaryType = Field(
        default=BoundaryType.START, description="Start or end of period"
    )
    start_monday: bool = Field(
        default=True,
        description="For week period: Consider Monday as start of week (False for Sunday)",
    )

    async def process(self, context: ProcessingContext) -> Datetime:
        dt = self.input_datetime.to_datetime()

        if self.period == PeriodType.DAY:
            if self.boundary == BoundaryType.START:
                return Datetime.from_datetime(
                    dt.replace(hour=0, minute=0, second=0, microsecond=0)
                )
            else:  # END
                return Datetime.from_datetime(
                    dt.replace(hour=23, minute=59, second=59, microsecond=999999)
                )

        elif self.period == PeriodType.WEEK:
            weekday = dt.weekday() if self.start_monday else (dt.weekday() + 1) % 7
            if self.boundary == BoundaryType.START:
                return Datetime.from_datetime(
                    (dt - timedelta(days=weekday)).replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                )
            else:  # END
                return Datetime.from_datetime(
                    (dt + timedelta(days=6 - weekday)).replace(
                        hour=23, minute=59, second=59, microsecond=999999
                    )
                )

        elif self.period == PeriodType.MONTH:
            if self.boundary == BoundaryType.START:
                return Datetime.from_datetime(
                    dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                )
            else:  # END
                # Get first day of next month and subtract one day
                if dt.month == 12:
                    next_month = dt.replace(year=dt.year + 1, month=1, day=1)
                else:
                    next_month = dt.replace(month=dt.month + 1, day=1)
                return Datetime.from_datetime(
                    next_month.replace(
                        hour=23, minute=59, second=59, microsecond=999999
                    )
                    - timedelta(days=1)
                )

        else:  # PeriodType.YEAR
            if self.boundary == BoundaryType.START:
                return Datetime.from_datetime(
                    dt.replace(
                        month=1, day=1, hour=0, minute=0, second=0, microsecond=0
                    )
                )
            else:  # END
                return Datetime.from_datetime(
                    dt.replace(
                        month=12,
                        day=31,
                        hour=23,
                        minute=59,
                        second=59,
                        microsecond=999999,
                    )
                )
