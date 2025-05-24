import asyncio
from datetime import datetime
import requests
from pydantic import Field
from typing import Literal

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import Datetime, BaseType
from nodetool.common.environment import Environment


CALENDLY_API_BASE = "https://api.calendly.com"


class CalendlyEvent(BaseType):
    """Represents a Calendly scheduled event."""

    type: Literal["calendly_event"] = "calendly_event"
    uri: str = ""
    name: str = ""
    start_time: Datetime = Datetime()
    end_time: Datetime = Datetime()
    location: str = ""


class ListScheduledEvents(BaseNode):
    """Fetch scheduled events for a Calendly user."""

    user: str = Field(
        default="",
        description="User URI to fetch events for",
    )
    count: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of events to return",
    )
    status: str = Field(
        default="active",
        description="Event status filter",
    )

    async def process(self, context: ProcessingContext) -> list[CalendlyEvent]:
        env = Environment.get_environment()
        token = env.get("CALENDLY_API_TOKEN") or env.get("CALENDLY_TOKEN")
        if not token:
            raise ValueError("CALENDLY_API_TOKEN is not set")

        params = {
            "user": self.user,
            "count": self.count,
            "status": self.status,
            "sort": "start_time:asc",
        }
        headers = {"Authorization": f"Bearer {token}"}
        res = await asyncio.to_thread(
            requests.get,
            f"{CALENDLY_API_BASE}/scheduled_events",
            params=params,
            headers=headers,
            timeout=10,
        )
        res.raise_for_status()
        data = res.json().get("collection", [])
        events = []
        for e in data:
            events.append(
                CalendlyEvent(
                    uri=e.get("uri", ""),
                    name=e.get("name", ""),
                    start_time=Datetime.from_datetime(
                        datetime.fromisoformat(e.get("start_time"))
                    ),
                    end_time=Datetime.from_datetime(
                        datetime.fromisoformat(e.get("end_time"))
                    ),
                    location=e.get("location", {}).get("location", ""),
                )
            )
        return events


class ScheduledEventFields(BaseNode):
    """Extract fields from a CalendlyEvent."""

    event: CalendlyEvent = Field(
        default=CalendlyEvent(), description="The Calendly event to extract"
    )

    @classmethod
    def return_type(cls):
        return {
            "uri": str,
            "name": str,
            "start_time": Datetime,
            "end_time": Datetime,
            "location": str,
        }

    async def process(self, context: ProcessingContext):
        return {
            "uri": self.event.uri,
            "name": self.event.name,
            "start_time": self.event.start_time,
            "end_time": self.event.end_time,
            "location": self.event.location,
        }
