"""
Extended Google search nodes for Nodetool.
Provides nodes for Google Events, Flights, Hotels, Patents, Videos, and Autocomplete via SerpAPI.
"""

from pydantic import Field
from typing import Any, TypedDict

from nodetool.nodes.search._base import SerpNode, format_results
from nodetool.workflows.processing_context import ProcessingContext


def _format_flights(flights: list[dict]) -> str:
    """Format flight results into human-readable text."""
    lines = []
    for i, flight_option in enumerate(flights):
        lines.append(f"--- Option {i + 1} ---")
        price = flight_option.get("price")
        if price:
            lines.append(f"  Price: {price}")
        total_duration = flight_option.get("total_duration")
        if total_duration:
            hours = total_duration // 60
            mins = total_duration % 60
            lines.append(f"  Total Duration: {hours}h {mins}m")
        flight_type = flight_option.get("type")
        if flight_type:
            lines.append(f"  Type: {flight_type}")
        carbon = flight_option.get("carbon_emissions", {})
        if carbon.get("this_flight"):
            lines.append(f"  CO2 Emissions: {carbon['this_flight']}g")
        segments = flight_option.get("flights", [])
        for j, seg in enumerate(segments):
            dep = seg.get("departure_airport", {})
            arr = seg.get("arrival_airport", {})
            lines.append(
                f"  Leg {j + 1}: {dep.get('name', '?')} ({dep.get('id', '?')}) {dep.get('time', '?')}"
                f" -> {arr.get('name', '?')} ({arr.get('id', '?')}) {arr.get('time', '?')}"
            )
            if seg.get("airline"):
                lines.append(f"    Airline: {seg['airline']} {seg.get('flight_number', '')}")
            if seg.get("duration"):
                lines.append(f"    Duration: {seg['duration']} min")
            if seg.get("travel_class"):
                lines.append(f"    Class: {seg['travel_class']}")
        layovers = flight_option.get("layovers", [])
        for lay in layovers:
            lines.append(
                f"  Layover: {lay.get('name', '?')} ({lay.get('id', '?')}) - {lay.get('duration', '?')} min"
            )
        lines.append("")
    return "\n".join(lines)


class GoogleEvents(SerpNode):
    """
    Search Google Events for local events, concerts, festivals, and activities.
    google, events, concerts, festivals, activities, local
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    query: str = Field(default="", description="Event search query (e.g., 'concerts in Austin')")
    location: str = Field(default="", description="Location for event search")
    num_results: int = Field(default=10, description="Maximum number of results")

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required")

        params: dict[str, Any] = {"q": self.query}
        if self.location:
            params["location"] = self.location

        result_data = await self._search_raw(context, "google_events", params)

        results = result_data.get("events_results", [])
        text = format_results(
            results,
            [
                ("date", "Date"),
                ("address", "Address"),
                ("link", None),
                ("description", None),
                ("venue", "Venue"),
            ],
        )
        return {"results": results, "text": text}


class GoogleFlights(SerpNode):
    """
    Search Google Flights for airline tickets, prices, and flight options.
    google, flights, airline, tickets, travel, booking
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    departure_id: str = Field(
        default="", description="Departure airport code (e.g., 'JFK', 'LAX') or location kgmid"
    )
    arrival_id: str = Field(
        default="", description="Arrival airport code (e.g., 'LHR', 'CDG') or location kgmid"
    )
    outbound_date: str = Field(
        default="", description="Departure date in YYYY-MM-DD format"
    )
    return_date: str = Field(
        default="",
        description="Return date in YYYY-MM-DD format (leave empty for one-way)",
    )
    trip_type: int = Field(
        default=1,
        description="Trip type: 1 = Round trip, 2 = One way, 3 = Multi-city",
    )
    adults: int = Field(default=1, description="Number of adult passengers")
    travel_class: int = Field(
        default=1,
        description="Cabin class: 1 = Economy, 2 = Premium economy, 3 = Business, 4 = First",
    )
    currency: str = Field(
        default="USD", description="Currency code for prices (e.g., 'USD', 'EUR', 'GBP')"
    )
    stops: int = Field(
        default=0,
        description="Maximum number of stops: 0 = any, 1 = non-stop, 2 = up to 1 stop, 3 = up to 2 stops",
    )
    sort_by: int = Field(
        default=1,
        description="Sort order: 1 = Top, 2 = Price, 3 = Departure, 4 = Arrival, 5 = Duration, 6 = Emissions",
    )

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.departure_id or not self.arrival_id:
            raise ValueError("Both departure_id and arrival_id are required")
        if not self.outbound_date:
            raise ValueError("Outbound date is required")
        if self.trip_type == 1 and not self.return_date:
            raise ValueError("Return date is required for round trip flights")

        params: dict[str, Any] = {
            "departure_id": self.departure_id,
            "arrival_id": self.arrival_id,
            "outbound_date": self.outbound_date,
            "type": str(self.trip_type),
            "adults": str(self.adults),
            "travel_class": str(self.travel_class),
            "currency": self.currency,
        }
        if self.return_date:
            params["return_date"] = self.return_date
        if self.stops:
            params["stops"] = str(self.stops)
        if self.sort_by != 1:
            params["sort_by"] = str(self.sort_by)

        result_data = await self._search_raw(context, "google_flights", params)

        best = result_data.get("best_flights", [])
        other = result_data.get("other_flights", [])
        all_flights = best + other
        text_parts = []
        if best:
            text_parts.append("=== Best Flights ===\n" + _format_flights(best))
        if other:
            text_parts.append("=== Other Flights ===\n" + _format_flights(other))
        price_insights = result_data.get("price_insights")
        if price_insights:
            lowest = price_insights.get("lowest_price")
            typical = price_insights.get("typical_price_range")
            if lowest:
                text_parts.append(f"Lowest Price: {lowest}")
            if typical:
                text_parts.append(f"Typical Range: {typical}")
        text = "\n".join(text_parts) if text_parts else "No flights found."
        return {"results": all_flights, "text": text}


class GoogleHotels(SerpNode):
    """
    Search Google Hotels for accommodations, pricing, and availability.
    google, hotels, accommodation, travel, booking, lodging
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    query: str = Field(
        default="", description="Hotel search query (e.g., 'hotels in Paris')"
    )
    check_in_date: str = Field(
        default="", description="Check-in date in YYYY-MM-DD format"
    )
    check_out_date: str = Field(
        default="", description="Check-out date in YYYY-MM-DD format"
    )
    adults: int = Field(default=2, description="Number of adults")
    children: int = Field(default=0, description="Number of children")
    currency: str = Field(
        default="USD", description="Currency code for prices (e.g., 'USD', 'EUR', 'GBP')"
    )
    sort_by: int = Field(
        default=0,
        description="Sort order: 0 = default, 3 = lowest price, 8 = highest rating, 13 = most reviewed",
    )
    min_price: int = Field(default=0, description="Minimum price per night filter")
    max_price: int = Field(default=0, description="Maximum price per night filter")
    hotel_class: int = Field(
        default=0, description="Minimum star rating: 2-5 (0 = any)"
    )

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required")
        if not self.check_in_date or not self.check_out_date:
            raise ValueError("Both check_in_date and check_out_date are required")

        params: dict[str, Any] = {
            "q": self.query,
            "check_in_date": self.check_in_date,
            "check_out_date": self.check_out_date,
            "adults": str(self.adults),
            "currency": self.currency,
        }
        if self.children > 0:
            params["children"] = str(self.children)
        if self.sort_by:
            params["sort_by"] = str(self.sort_by)
        if self.min_price > 0:
            params["min_price"] = str(self.min_price)
        if self.max_price > 0:
            params["max_price"] = str(self.max_price)
        if self.hotel_class >= 2:
            params["hotel_class"] = str(self.hotel_class)

        result_data = await self._search_raw(context, "google_hotels", params)

        results = result_data.get("properties", [])
        lines = []
        for i, prop in enumerate(results):
            name = prop.get("name", "Untitled")
            lines.append(f"[{i + 1}] {name}")
            prop_type = prop.get("type")
            if prop_type:
                lines.append(f"    Type: {prop_type}")
            hotel_cls = prop.get("hotel_class") or prop.get("extracted_hotel_class")
            if hotel_cls:
                lines.append(f"    Class: {hotel_cls}-star")
            rating = prop.get("overall_rating")
            if rating is not None:
                rating_str = f"    Rating: {rating}"
                reviews = prop.get("reviews")
                if reviews is not None:
                    rating_str += f" ({reviews} reviews)"
                lines.append(rating_str)
            rate = prop.get("rate_per_night", {})
            lowest = rate.get("lowest")
            if lowest:
                lines.append(f"    Price: {lowest}/night")
            total = prop.get("total_rate", {})
            total_lowest = total.get("lowest")
            if total_lowest:
                lines.append(f"    Total: {total_lowest}")
            desc = prop.get("description")
            if desc:
                lines.append(f"    {desc[:200]}")
            check_in = prop.get("check_in_time")
            check_out = prop.get("check_out_time")
            if check_in or check_out:
                lines.append(f"    Check-in: {check_in or 'N/A'} / Check-out: {check_out or 'N/A'}")
            lines.append("")
        text = "\n".join(lines) if lines else "No hotels found."
        return {"results": results, "text": text}


class GooglePatents(SerpNode):
    """
    Search Google Patents for patent documents, filings, and prior art.
    google, patents, intellectual property, inventions, prior art, filings
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    query: str = Field(
        default="", description="Patent search query (keywords, patent numbers, or concepts)"
    )
    num_results: int = Field(
        default=10, description="Number of results per page (10-100)"
    )
    sort: str = Field(
        default="", description="Sort order: 'new' for newest first, 'old' for oldest first, empty for relevance"
    )
    inventor: str = Field(
        default="", description="Filter by inventor name(s), comma-separated"
    )
    assignee: str = Field(
        default="", description="Filter by assignee/company name(s), comma-separated"
    )
    country: str = Field(
        default="", description="Filter by country code(s), comma-separated (e.g., 'US,EP,WO')"
    )
    status: str = Field(
        default="", description="Filter by status: 'GRANT' or 'APPLICATION'"
    )
    language: str = Field(
        default="", description="Filter by language (e.g., 'ENGLISH', 'GERMAN', 'CHINESE')"
    )
    before_date: str = Field(
        default="",
        description="Maximum date filter in format type:YYYYMMDD (e.g., 'publication:20230101')",
    )
    after_date: str = Field(
        default="",
        description="Minimum date filter in format type:YYYYMMDD (e.g., 'filing:20200101')",
    )

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required")

        params: dict[str, Any] = {
            "q": self.query,
            "num": str(self.num_results),
        }
        if self.sort:
            params["sort"] = self.sort
        if self.inventor:
            params["inventor"] = self.inventor
        if self.assignee:
            params["assignee"] = self.assignee
        if self.country:
            params["country"] = self.country
        if self.status:
            params["status"] = self.status
        if self.language:
            params["language"] = self.language
        if self.before_date:
            params["before"] = self.before_date
        if self.after_date:
            params["after"] = self.after_date

        result_data = await self._search_raw(context, "google_patents", params)

        results = result_data.get("organic_results", [])
        lines = []
        for i, r in enumerate(results):
            pos = r.get("position", i + 1)
            title = r.get("title", "Untitled")
            lines.append(f"[{pos}] {title}")
            patent_id = r.get("patent_id")
            if patent_id:
                lines.append(f"    Patent ID: {patent_id}")
            assignee = r.get("assignee")
            if assignee:
                lines.append(f"    Assignee: {assignee}")
            inventor = r.get("inventor")
            if inventor:
                lines.append(f"    Inventor: {inventor}")
            filing_date = r.get("filing_date")
            if filing_date:
                lines.append(f"    Filing Date: {filing_date}")
            grant_date = r.get("grant_date")
            if grant_date:
                lines.append(f"    Grant Date: {grant_date}")
            priority_date = r.get("priority_date")
            if priority_date:
                lines.append(f"    Priority Date: {priority_date}")
            snippet = r.get("snippet")
            if snippet:
                lines.append(f"    {snippet[:200]}")
            link = r.get("patent_link")
            if link:
                lines.append(f"    {link}")
            lines.append("")
        text = "\n".join(lines) if lines else "No patents found."
        return {"results": results, "text": text}


class GoogleVideos(SerpNode):
    """
    Search Google Videos for video content across the web.
    google, videos, search, media, clips, streaming
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    query: str = Field(
        default="", description="Video search query"
    )
    num_results: int = Field(
        default=10, description="Maximum number of video results (controls pagination via start offset)"
    )
    location: str = Field(
        default="", description="Geographic location for search"
    )

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required")

        params: dict[str, Any] = {"q": self.query}
        if self.location:
            params["location"] = self.location

        result_data = await self._search_raw(context, "google_videos", params)

        results = result_data.get("video_results", [])
        text = format_results(
            results,
            [
                ("link", None),
                ("displayed_link", "Source"),
                ("duration", "Duration"),
                ("date", "Date"),
                ("snippet", None),
            ],
        )
        return {"results": results, "text": text}


class GoogleAutocomplete(SerpNode):
    """
    Get Google Autocomplete suggestions for a search query.
    google, autocomplete, suggestions, search, typeahead, completion
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    query: str = Field(
        default="", description="Partial search query to get completions for"
    )
    language: str = Field(
        default="", description="Two-letter language code (e.g., 'en', 'de', 'fr')"
    )
    country: str = Field(
        default="", description="Two-letter country code (e.g., 'us', 'de', 'fr')"
    )

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required")

        params: dict[str, Any] = {"q": self.query}
        if self.language:
            params["hl"] = self.language
        if self.country:
            params["gl"] = self.country

        result_data = await self._search_raw(context, "google_autocomplete", params)

        suggestions = result_data.get("suggestions", [])
        lines = []
        for i, s in enumerate(suggestions):
            value = s.get("value", "")
            relevance = s.get("relevance")
            line = f"[{i + 1}] {value}"
            if relevance is not None:
                line += f" (relevance: {relevance})"
            lines.append(line)
        text = "\n".join(lines) if lines else "No suggestions found."
        return {"results": suggestions, "text": text}
