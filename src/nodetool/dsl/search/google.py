from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class GoogleFinance(GraphNode):
    """
    Retrieve financial market data from Google Finance.
    google, finance, stocks, market, serp
    """

    query: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Stock symbol or company name to search for"
    )
    window: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="Time window for financial data (e.g., '1d', '5d', '1m', '3m', '6m', '1y', '5y')",
    )

    @classmethod
    def get_node_type(cls):
        return "search.google.GoogleFinance"


class GoogleImages(GraphNode):
    """
    Search Google Images to retrieve live image results.
    google, images, serp, visual, reverse, search
    """

    keyword: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Search query or keyword for images"
    )
    image_url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="URL of image for reverse image search"
    )
    num_results: int | GraphNode | tuple[GraphNode, str] = Field(
        default=20, description="Maximum number of image results to return"
    )

    @classmethod
    def get_node_type(cls):
        return "search.google.GoogleImages"


class GoogleJobs(GraphNode):
    """
    Search Google Jobs for job listings.
    google, jobs, employment, careers, serp
    """

    query: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Job title, skills, or company name to search for"
    )
    location: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Geographic location for job search"
    )
    num_results: int | GraphNode | tuple[GraphNode, str] = Field(
        default=10, description="Maximum number of job results to return"
    )

    @classmethod
    def get_node_type(cls):
        return "search.google.GoogleJobs"


class GoogleLens(GraphNode):
    """
    Search with an image URL using Google Lens to find visual matches and related content.
    google, lens, visual, image, search, serp
    """

    image_url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="URL of the image to analyze with Google Lens"
    )
    num_results: int | GraphNode | tuple[GraphNode, str] = Field(
        default=10, description="Maximum number of visual search results to return"
    )

    @classmethod
    def get_node_type(cls):
        return "search.google.GoogleLens"


class GoogleMaps(GraphNode):
    """
    Search Google Maps for places or get details about a specific place.
    google, maps, places, locations, serp
    """

    query: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Place name, address, or location query"
    )
    num_results: int | GraphNode | tuple[GraphNode, str] = Field(
        default=10, description="Maximum number of map results to return"
    )

    @classmethod
    def get_node_type(cls):
        return "search.google.GoogleMaps"


class GoogleNews(GraphNode):
    """
    Search Google News to retrieve live news articles.
    google, news, serp, articles
    """

    keyword: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Search query or keyword for news articles"
    )
    num_results: int | GraphNode | tuple[GraphNode, str] = Field(
        default=10, description="Maximum number of news results to return"
    )

    @classmethod
    def get_node_type(cls):
        return "search.google.GoogleNews"


class GoogleSearch(GraphNode):
    """
    Search Google to retrieve organic search results.
    google, search, serp, web
    """

    keyword: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Search query or keyword to search for"
    )
    num_results: int | GraphNode | tuple[GraphNode, str] = Field(
        default=10, description="Maximum number of results to return"
    )

    @classmethod
    def get_node_type(cls):
        return "search.google.GoogleSearch"


class GoogleShopping(GraphNode):
    """
    Search Google Shopping for products.
    google, shopping, products, ecommerce, serp
    """

    query: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Product name or description to search for"
    )
    country: str | GraphNode | tuple[GraphNode, str] = Field(
        default="us",
        description="Country code for shopping search (e.g., 'us', 'uk', 'ca')",
    )
    min_price: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="Minimum price filter for products"
    )
    max_price: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="Maximum price filter for products"
    )
    condition: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="Product condition filter (e.g., 'new', 'used', 'refurbished')",
    )
    sort_by: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="Sort order for results (e.g., 'price_low_to_high', 'price_high_to_low', 'review_score')",
    )
    num_results: int | GraphNode | tuple[GraphNode, str] = Field(
        default=10, description="Maximum number of shopping results to return"
    )

    @classmethod
    def get_node_type(cls):
        return "search.google.GoogleShopping"
