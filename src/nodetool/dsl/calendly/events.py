from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class ListScheduledEvents(GraphNode):
    """Fetch scheduled events for a Calendly user."""

    user: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="User URI to fetch events for"
    )
    count: int | GraphNode | tuple[GraphNode, str] = Field(
        default=20, description="Number of events to return"
    )
    status: str | GraphNode | tuple[GraphNode, str] = Field(
        default="active", description="Event status filter"
    )

    @classmethod
    def get_node_type(cls):
        return "calendly.events.ListScheduledEvents"


class ScheduledEventFields(GraphNode):
    """Extract fields from a CalendlyEvent."""

    event: types.CalendlyEvent | GraphNode | tuple[GraphNode, str] = Field(
        default=types.CalendlyEvent(
            type="calendly_event",
            uri="",
            name="",
            start_time=types.Datetime(
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
            end_time=types.Datetime(
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
            location="",
        ),
        description="The Calendly event to extract",
    )

    @classmethod
    def get_node_type(cls):
        return "calendly.events.ScheduledEventFields"
