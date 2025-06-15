from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class AddTimeDelta(GraphNode):
    """
    Add or subtract time from a datetime.
    datetime, add, subtract

    Use cases:
    - Calculate future/past dates
    - Generate date ranges
    """

    input_datetime: types.Datetime | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Datetime(
            type="datetime",
            year=0,
            month=0,
            day=0,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo="UTC",
            utc_offset=0,
        ),
        description="Starting datetime",
    )
    days: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="Number of days to add (negative to subtract)"
    )
    hours: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="Number of hours to add (negative to subtract)"
    )
    minutes: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="Number of minutes to add (negative to subtract)"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.date.AddTimeDelta"


class DateDifference(GraphNode):
    """
    Calculate the difference between two dates.
    datetime, difference, duration

    Use cases:
    - Calculate time periods
    - Measure durations
    """

    start_date: types.Datetime | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Datetime(
            type="datetime",
            year=0,
            month=0,
            day=0,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo="UTC",
            utc_offset=0,
        ),
        description="Start datetime",
    )
    end_date: types.Datetime | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Datetime(
            type="datetime",
            year=0,
            month=0,
            day=0,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo="UTC",
            utc_offset=0,
        ),
        description="End datetime",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.date.DateDifference"


class DateRange(GraphNode):
    """
    Generate a list of dates between start and end dates.
    datetime, range, list

    Use cases:
    - Generate date sequences
    - Create date-based iterations
    """

    start_date: types.Datetime | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Datetime(
            type="datetime",
            year=0,
            month=0,
            day=0,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo="UTC",
            utc_offset=0,
        ),
        description="Start date of the range",
    )
    end_date: types.Datetime | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Datetime(
            type="datetime",
            year=0,
            month=0,
            day=0,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo="UTC",
            utc_offset=0,
        ),
        description="End date of the range",
    )
    step_days: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1, description="Number of days between each date"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.date.DateRange"


class DateToDatetime(GraphNode):
    """
    Convert a Date object to a Datetime object.
    date, datetime, convert
    """

    input_date: types.Date | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Date(type="date", year=0, month=0, day=0),
        description="Date to convert",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.date.DateToDatetime"


class DatetimeToDate(GraphNode):
    """
    Convert a Datetime object to a Date object.
    date, datetime, convert
    """

    input_datetime: types.Datetime | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Datetime(
            type="datetime",
            year=0,
            month=0,
            day=0,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo="UTC",
            utc_offset=0,
        ),
        description="Datetime to convert",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.date.DatetimeToDate"


class DaysAgo(GraphNode):
    """
    Get datetime from specified days ago.
    datetime, past, days
    """

    days: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1, description="Number of days ago"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.date.DaysAgo"


class DaysFromNow(GraphNode):
    """
    Get datetime specified days in the future.
    datetime, future, days
    """

    days: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1, description="Number of days in the future"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.date.DaysFromNow"


class EndOfDay(GraphNode):
    """
    Get the datetime set to the end of the day (23:59:59).
    datetime, day, end
    """

    input_datetime: types.Datetime | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Datetime(
            type="datetime",
            year=0,
            month=0,
            day=0,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo="UTC",
            utc_offset=0,
        ),
        description="Input datetime",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.date.EndOfDay"


class EndOfMonth(GraphNode):
    """
    Get the datetime set to the last day of the month.
    datetime, month, end
    """

    input_datetime: types.Datetime | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Datetime(
            type="datetime",
            year=0,
            month=0,
            day=0,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo="UTC",
            utc_offset=0,
        ),
        description="Input datetime",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.date.EndOfMonth"


class EndOfWeek(GraphNode):
    """
    Get the datetime set to the last day of the week (Sunday by default).
    datetime, week, end
    """

    input_datetime: types.Datetime | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Datetime(
            type="datetime",
            year=0,
            month=0,
            day=0,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo="UTC",
            utc_offset=0,
        ),
        description="Input datetime",
    )
    start_monday: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Consider Monday as start of week (False for Sunday)"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.date.EndOfWeek"


class EndOfYear(GraphNode):
    """
    Get the datetime set to the last day of the year.
    datetime, year, end
    """

    input_datetime: types.Datetime | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Datetime(
            type="datetime",
            year=0,
            month=0,
            day=0,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo="UTC",
            utc_offset=0,
        ),
        description="Input datetime",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.date.EndOfYear"


import nodetool.nodes.lib.date


class FormatDateTime(GraphNode):
    """
    Convert a datetime object to a formatted string.
    datetime, format, convert

    Use cases:
    - Standardize date formats
    - Prepare dates for different systems
    """

    DateFormat: typing.ClassVar[type] = nodetool.nodes.lib.date.DateFormat
    input_datetime: types.Datetime | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Datetime(
            type="datetime",
            year=0,
            month=0,
            day=0,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo="UTC",
            utc_offset=0,
        ),
        description="Datetime object to format",
    )
    output_format: nodetool.nodes.lib.date.DateFormat = Field(
        default=nodetool.nodes.lib.date.DateFormat.HUMAN_READABLE,
        description="Desired output format",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.date.FormatDateTime"


class GetQuarter(GraphNode):
    """
    Get the quarter number and start/end dates for a given datetime.
    datetime, quarter, period

    Use cases:
    - Financial reporting periods
    - Quarterly analytics
    """

    input_datetime: types.Datetime | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Datetime(
            type="datetime",
            year=0,
            month=0,
            day=0,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo="UTC",
            utc_offset=0,
        ),
        description="Input datetime",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.date.GetQuarter"


class GetWeekday(GraphNode):
    """
    Get the weekday name or number from a datetime.
    datetime, weekday, name

    Use cases:
    - Get day names for scheduling
    - Filter events by weekday
    """

    input_datetime: types.Datetime | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Datetime(
            type="datetime",
            year=0,
            month=0,
            day=0,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo="UTC",
            utc_offset=0,
        ),
        description="Input datetime",
    )
    as_name: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Return weekday name instead of number (0-6)"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.date.GetWeekday"


class HoursAgo(GraphNode):
    """
    Get datetime from specified hours ago.
    datetime, past, hours
    """

    hours: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1, description="Number of hours ago"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.date.HoursAgo"


class HoursFromNow(GraphNode):
    """
    Get datetime specified hours in the future.
    datetime, future, hours
    """

    hours: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1, description="Number of hours in the future"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.date.HoursFromNow"


class IsDateInRange(GraphNode):
    """
    Check if a date falls within a specified range.
    datetime, range, check

    Use cases:
    - Validate date ranges
    - Filter date-based data
    """

    check_date: types.Datetime | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Datetime(
            type="datetime",
            year=0,
            month=0,
            day=0,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo="UTC",
            utc_offset=0,
        ),
        description="Date to check",
    )
    start_date: types.Datetime | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Datetime(
            type="datetime",
            year=0,
            month=0,
            day=0,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo="UTC",
            utc_offset=0,
        ),
        description="Start of date range",
    )
    end_date: types.Datetime | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Datetime(
            type="datetime",
            year=0,
            month=0,
            day=0,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo="UTC",
            utc_offset=0,
        ),
        description="End of date range",
    )
    inclusive: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Include start and end dates in range"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.date.IsDateInRange"


class MonthsAgo(GraphNode):
    """
    Get datetime from specified months ago.
    datetime, past, months
    """

    months: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1, description="Number of months ago"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.date.MonthsAgo"


class MonthsFromNow(GraphNode):
    """
    Get datetime specified months in the future.
    datetime, future, months
    """

    months: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1, description="Number of months in the future"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.date.MonthsFromNow"


class Now(GraphNode):
    """
    Get the current date and time.
    datetime, current, now
    """

    @classmethod
    def get_node_type(cls):
        return "lib.date.Now"


import nodetool.nodes.lib.date


class ParseDate(GraphNode):
    """
    Parse a date string into components.
    date, parse, format
    """

    DateFormat: typing.ClassVar[type] = nodetool.nodes.lib.date.DateFormat
    date_string: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The date string to parse"
    )
    input_format: nodetool.nodes.lib.date.DateFormat = Field(
        default=nodetool.nodes.lib.date.DateFormat.ISO,
        description="Format of the input date string",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.date.ParseDate"


import nodetool.nodes.lib.date


class ParseDateTime(GraphNode):
    """
    Parse a date/time string into components.
    datetime, parse, format

    Use cases:
    - Extract date components from strings
    - Convert between date formats
    """

    DateFormat: typing.ClassVar[type] = nodetool.nodes.lib.date.DateFormat
    datetime_string: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The datetime string to parse"
    )
    input_format: nodetool.nodes.lib.date.DateFormat = Field(
        default=nodetool.nodes.lib.date.DateFormat.ISO,
        description="Format of the input datetime string",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.date.ParseDateTime"


class StartOfDay(GraphNode):
    """
    Get the datetime set to the start of the day (00:00:00).
    datetime, day, start
    """

    input_datetime: types.Datetime | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Datetime(
            type="datetime",
            year=0,
            month=0,
            day=0,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo="UTC",
            utc_offset=0,
        ),
        description="Input datetime",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.date.StartOfDay"


class StartOfMonth(GraphNode):
    """
    Get the datetime set to the first day of the month.
    datetime, month, start
    """

    input_datetime: types.Datetime | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Datetime(
            type="datetime",
            year=0,
            month=0,
            day=0,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo="UTC",
            utc_offset=0,
        ),
        description="Input datetime",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.date.StartOfMonth"


class StartOfWeek(GraphNode):
    """
    Get the datetime set to the first day of the week (Monday by default).
    datetime, week, start
    """

    input_datetime: types.Datetime | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Datetime(
            type="datetime",
            year=0,
            month=0,
            day=0,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo="UTC",
            utc_offset=0,
        ),
        description="Input datetime",
    )
    start_monday: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Consider Monday as start of week (False for Sunday)"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.date.StartOfWeek"


class StartOfYear(GraphNode):
    """
    Get the datetime set to the first day of the year.
    datetime, year, start
    """

    input_datetime: types.Datetime | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Datetime(
            type="datetime",
            year=0,
            month=0,
            day=0,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo="UTC",
            utc_offset=0,
        ),
        description="Input datetime",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.date.StartOfYear"


class Today(GraphNode):
    """
    Get the current date.
    date, today, now
    """

    @classmethod
    def get_node_type(cls):
        return "lib.date.Today"
