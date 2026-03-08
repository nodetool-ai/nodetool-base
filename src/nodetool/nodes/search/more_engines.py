"""
Additional SerpAPI engine nodes for Nodetool.
Provides Google Maps Reviews, Bing Shopping, Bing Videos, and YouTube Transcript nodes.
"""

from enum import Enum

from pydantic import Field
from typing import TypedDict

from nodetool.nodes.search._base import SerpNode, format_results
from nodetool.workflows.processing_context import ProcessingContext


# ── Google Maps Reviews ──────────────────────────────────────────────


class ReviewSortOrder(str, Enum):
    MOST_RELEVANT = "qualityScore"
    NEWEST = "newestFirst"
    HIGHEST_RATING = "ratingHigh"
    LOWEST_RATING = "ratingLow"


class GoogleMapsReviews(SerpNode):
    """
    Retrieve reviews for a Google Maps place by its data_id.
    Use Google Maps search to find a place's data_id first.
    google, maps, reviews, places, ratings, local, business, restaurant
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    data_id: str = Field(
        default="",
        description="Google Maps place data_id (obtain from a Google Maps search result)",
    )
    sort_by: ReviewSortOrder = Field(
        default=ReviewSortOrder.MOST_RELEVANT,
        description="Sort order for reviews",
    )
    num_results: int = Field(
        default=10,
        description="Number of reviews to return (1-20)",
        ge=1,
        le=20,
    )

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.data_id:
            raise ValueError("data_id is required. Use a Google Maps search to find the place data_id.")

        result_data = await self._search_raw(
            context,
            "google_maps_reviews",
            {
                "data_id": self.data_id,
                "sort_by": self.sort_by.value,
                "num": self.num_results,
            },
        )

        reviews = result_data.get("reviews", [])

        lines: list[str] = []
        for i, review in enumerate(reviews):
            user = review.get("user", {})
            name = user.get("name", "Anonymous") if isinstance(user, dict) else "Anonymous"
            rating = review.get("rating", "N/A")
            lines.append(f"[{i + 1}] {name} - {rating}/5")

            date = review.get("date") or review.get("iso_date", "")
            if date:
                lines.append(f"    Date: {date}")

            snippet = review.get("snippet", "")
            if snippet:
                lines.append(f"    {snippet}")

            source = review.get("source", "")
            if source:
                lines.append(f"    Source: {source}")

            lines.append("")

        place_info = result_data.get("place_info", {})
        header = ""
        if place_info:
            title = place_info.get("title", "")
            overall_rating = place_info.get("rating", "")
            total_reviews = place_info.get("reviews", "")
            if title:
                header = f"Place: {title}"
                if overall_rating:
                    header += f" | Rating: {overall_rating}"
                if total_reviews:
                    header += f" | Total reviews: {total_reviews}"
                header += "\n\n"

        return {"results": reviews, "text": header + "\n".join(lines)}


# ── Bing Shopping ────────────────────────────────────────────────────


class BingShopping(SerpNode):
    """
    Search Bing Shopping for product listings with prices and seller info.
    bing, shopping, products, prices, ecommerce, deals, buy
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    query: str = Field(default="", description="Product search query")
    num_results: int = Field(
        default=10, description="Maximum number of shopping results to return"
    )

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required")

        result_data = await self._search_raw(
            context,
            "bing_shopping",
            {
                "q": self.query,
                "count": self.num_results,
            },
        )

        results = result_data.get("shopping_results", [])
        text = format_results(
            results,
            [
                ("price", "Price"),
                ("seller", "Seller"),
                ("source", "Source"),
                ("rating", "Rating"),
                ("reviews", "Reviews"),
                ("link", None),
            ],
        )
        return {"results": results, "text": text}


# ── Bing Videos ──────────────────────────────────────────────────────


class BingVideos(SerpNode):
    """
    Search Bing Videos for video content across the web.
    bing, videos, search, media, streaming, content
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    query: str = Field(default="", description="Video search query")
    num_results: int = Field(
        default=10, description="Maximum number of video results to return"
    )

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required")

        result_data = await self._search_raw(
            context,
            "bing_videos",
            {
                "q": self.query,
                "count": self.num_results,
            },
        )

        results = result_data.get("video_results", [])
        text = format_results(
            results,
            [
                ("source", "Source"),
                ("channel", "Channel"),
                ("length", "Duration"),
                ("views", "Views"),
                ("date", "Date"),
                ("link", None),
            ],
        )
        return {"results": results, "text": text}


# ── YouTube Transcript ───────────────────────────────────────────────


class YouTubeTranscript(SerpNode):
    """
    Retrieve the full transcript of a YouTube video by its video ID.
    Concatenates all transcript segments into a single text output.
    youtube, transcript, subtitles, captions, text, video, speech
    """

    class OutputType(TypedDict):
        text: str
        segments: list[dict]

    video_id: str = Field(
        default="",
        description="YouTube video ID (the part after v= in the URL)",
    )
    lang: str = Field(
        default="en",
        description="Language code for the transcript (e.g., 'en', 'es', 'de')",
    )

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.video_id:
            raise ValueError("video_id is required")

        result_data = await self._search_raw(
            context,
            "youtube_video_transcript",
            {
                "v": self.video_id,
                "language_code": self.lang,
            },
        )

        # Transcript segments are returned under "transcript"
        segments = result_data.get("transcript", [])

        # Concatenate all snippet texts into a single transcript
        texts: list[str] = []
        for segment in segments:
            snippet = segment.get("snippet", "")
            if snippet:
                texts.append(snippet.strip())

        full_text = " ".join(texts)

        return {"text": full_text, "segments": segments}
