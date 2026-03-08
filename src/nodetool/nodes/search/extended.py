"""
Extended search nodes for Nodetool.
Provides additional nodes for YouTube Video details, DuckDuckGo News, Yelp Reviews, and Google Scholar Author.
"""

from pydantic import Field
from typing import Any, TypedDict

from nodetool.nodes.search._base import SerpNode
from nodetool.workflows.processing_context import ProcessingContext


class YouTubeVideo(SerpNode):
    """
    Get detailed information about a specific YouTube video by video ID.
    youtube, video, details, metadata, views, likes, channel
    """

    video_id: str = Field(
        default="",
        description="YouTube video ID (the 'v' parameter from a YouTube URL, e.g. 'dQw4w9WgXcQ')",
    )

    async def process(self, context: ProcessingContext) -> dict:
        if not self.video_id:
            raise ValueError("Video ID is required")

        params: dict[str, Any] = {
            "v": self.video_id,
        }

        result_data = await self._search_raw(context, "youtube_video", params)

        # The response contains video info at the top level
        # Extract the most relevant fields into a clean dict
        video_info: dict[str, Any] = {}
        for key in (
            "title", "thumbnail", "views", "extracted_views",
            "likes", "extracted_likes", "published_date",
            "live", "description", "comment_count",
            "extracted_comment_count",
        ):
            if key in result_data:
                video_info[key] = result_data[key]

        # Channel info
        channel = result_data.get("channel")
        if channel:
            video_info["channel"] = channel

        # Chapters
        chapters = result_data.get("chapters")
        if chapters:
            video_info["chapters"] = chapters

        # Related videos (just a summary)
        related = result_data.get("related_videos")
        if related:
            video_info["related_video_count"] = len(related)

        return video_info


class DuckDuckGoNews(SerpNode):
    """
    Search DuckDuckGo News for recent news articles.
    duckduckgo, news, search, articles, current events, media
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    query: str = Field(default="", description="News search query")
    num_results: int = Field(default=10, description="Maximum number of results to return")

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.query:
            raise ValueError("Query is required")

        params: dict[str, Any] = {
            "q": self.query,
        }

        result_data = await self._search_raw(context, "duckduckgo_news", params)

        results = result_data.get("news_results", [])
        results = results[: self.num_results]

        formatted_lines = []
        for i, r in enumerate(results):
            pos = r.get("position", i + 1)
            title = r.get("title", "Untitled")
            formatted_lines.append(f"[{pos}] {title}")
            if r.get("source"):
                formatted_lines.append(f"    Source: {r['source']}")
            if r.get("date"):
                formatted_lines.append(f"    Date: {r['date']}")
            if r.get("snippet"):
                formatted_lines.append(f"    {r['snippet']}")
            if r.get("link"):
                formatted_lines.append(f"    {r['link']}")
            formatted_lines.append("")

        return {"results": results, "text": "\n".join(formatted_lines)}


class YelpReviews(SerpNode):
    """
    Get reviews for a specific Yelp business by place ID.
    yelp, reviews, business, ratings, feedback, local
    """

    class OutputType(TypedDict):
        results: list[dict]
        text: str

    place_id: str = Field(
        default="",
        description="Yelp place ID (e.g. 'ED7A7vDdg8yLNKJTSVHHmg')",
    )
    num_results: int = Field(default=10, description="Maximum number of reviews to return")

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.place_id:
            raise ValueError("Place ID is required")

        params: dict[str, Any] = {
            "place_id": self.place_id,
        }

        result_data = await self._search_raw(context, "yelp_reviews", params)

        results = result_data.get("reviews", [])
        results = results[: self.num_results]

        formatted_lines = []
        for i, r in enumerate(results):
            pos = r.get("position", i + 1)
            user = r.get("user", {})
            user_name = user.get("name", "Anonymous") if isinstance(user, dict) else "Anonymous"
            formatted_lines.append(f"[{pos}] Review by {user_name}")
            if r.get("rating") is not None:
                formatted_lines.append(f"    Rating: {r['rating']}/5")
            if r.get("date"):
                formatted_lines.append(f"    Date: {r['date']}")
            comment = r.get("comment", {})
            if isinstance(comment, dict):
                text = comment.get("text", "")
            else:
                text = str(comment) if comment else ""
            if text:
                if len(text) > 200:
                    text = text[:197] + "..."
                formatted_lines.append(f"    {text}")
            if r.get("tags"):
                formatted_lines.append(f"    Tags: {', '.join(str(t) for t in r['tags'])}")
            owner_replies = r.get("owner_replies")
            if owner_replies:
                formatted_lines.append(f"    Owner replied: Yes")
            formatted_lines.append("")

        return {"results": results, "text": "\n".join(formatted_lines)}


class GoogleScholarAuthor(SerpNode):
    """
    Get detailed information about a Google Scholar author by author ID, including articles and citation metrics.
    google, scholar, author, academic, researcher, citations, h-index, publications
    """

    author_id: str = Field(
        default="",
        description="Google Scholar author ID (found in the profile URL)",
    )

    async def process(self, context: ProcessingContext) -> dict:
        if not self.author_id:
            raise ValueError("Author ID is required")

        params: dict[str, Any] = {
            "author_id": self.author_id,
        }

        result_data = await self._search_raw(context, "google_scholar_author", params)

        # Build a clean author profile dict
        author_profile: dict[str, Any] = {}

        # Author info
        author = result_data.get("author")
        if author:
            author_profile["author"] = author

        # Articles list
        articles = result_data.get("articles", [])
        if articles:
            author_profile["articles"] = articles

        # Citation metrics (h-index, i10-index, etc.)
        cited_by = result_data.get("cited_by")
        if cited_by:
            author_profile["cited_by"] = cited_by

        # Public access stats
        public_access = result_data.get("public_access")
        if public_access:
            author_profile["public_access"] = public_access

        # Co-authors
        co_authors = result_data.get("co_authors")
        if co_authors:
            author_profile["co_authors"] = co_authors

        return author_profile
