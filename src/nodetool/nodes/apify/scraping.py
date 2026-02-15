#!/usr/bin/env python

"""
Apify web scraping nodes for Nodetool.
Provides nodes for popular Apify actors including web scraping, social media, and e-commerce.
"""

from pydantic import Field
from typing import Any, Dict, List, ClassVar
from urllib.parse import quote

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

# Constants
DEFAULT_PAGE_FUNCTION = "async function pageFunction(context) { return context.request.loadedUrl; }"
MIN_RESULTS_PER_PAGE = 10
MAX_RESULTS_PER_PAGE = 100


async def _get_apify_client(context: ProcessingContext):
    """Get configured Apify client from context."""
    from apify_client import ApifyClient
    
    # Get API token from environment
    api_token = context.get_env("APIFY_API_TOKEN")
    if not api_token:
        return None, {"error": "APIFY_API_TOKEN not configured in environment"}
    
    client = ApifyClient(token=api_token)
    return client, None


class ApifyWebScraper(BaseNode):
    """
    Scrape websites using Apify's Web Scraper actor.
    Extracts data from web pages using CSS selectors or custom JavaScript.
    apify, scraping, web, data, extraction, crawler
    """

    start_urls: List[str] = Field(
        default_factory=list,
        description="List of URLs to scrape"
    )
    link_selector: str = Field(
        default="a[href]",
        description="CSS selector for links to follow"
    )
    page_function: str = Field(
        default="",
        description="JavaScript function to execute on each page"
    )
    max_pages: int = Field(
        default=10,
        description="Maximum number of pages to scrape"
    )
    wait_for_finish: int = Field(
        default=300,
        description="Maximum time to wait for scraping to complete (seconds)"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> List[Dict[str, Any]]:
        if not self.start_urls:
            raise ValueError("start_urls is required")

        client, error = await _get_apify_client(context)
        if error:
            raise ValueError(error.get("error", "Failed to configure Apify client"))

        # Prepare input for Web Scraper actor
        run_input = {
            "startUrls": [{"url": url} for url in self.start_urls],
            "linkSelector": self.link_selector,
            "pageFunction": self.page_function or DEFAULT_PAGE_FUNCTION,
            "maxPagesPerCrawl": self.max_pages,
        }

        # Run the Web Scraper actor
        actor = client.actor("apify/web-scraper")
        run = actor.call(run_input=run_input, wait_secs=self.wait_for_finish)

        if not run:
            return []

        # Get dataset results
        dataset_id = run.get("defaultDatasetId")
        if dataset_id:
            dataset = client.dataset(dataset_id)
            items = list(dataset.iterate_items())
            return items

        return []


class ApifyGoogleSearchScraper(BaseNode):
    """
    Scrape Google Search results using Apify's Google Search Scraper.
    Extract organic results, ads, related searches, and more.
    apify, google, search, serp, scraping, seo
    """

    queries: List[str] = Field(
        default_factory=list,
        description="List of search queries to execute"
    )
    country_code: str = Field(
        default="us",
        description="Country code for Google search (e.g., 'us', 'uk', 'de')"
    )
    language_code: str = Field(
        default="en",
        description="Language code for results (e.g., 'en', 'es', 'fr')"
    )
    max_pages: int = Field(
        default=1,
        description="Maximum number of result pages per query"
    )
    results_per_page: int = Field(
        default=100,
        description="Number of results per page (10-100)"
    )
    wait_for_finish: int = Field(
        default=300,
        description="Maximum time to wait for scraping to complete (seconds)"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> List[Dict[str, Any]]:
        if not self.queries:
            raise ValueError("queries is required")

        client, error = await _get_apify_client(context)
        if error:
            raise ValueError(error.get("error", "Failed to configure Apify client"))

        # Prepare input for Google Search Scraper
        run_input = {
            "queries": "\n".join(self.queries),
            "countryCode": self.country_code,
            "languageCode": self.language_code,
            "maxPagesPerQuery": self.max_pages,
            "resultsPerPage": min(max(MIN_RESULTS_PER_PAGE, self.results_per_page), MAX_RESULTS_PER_PAGE),
        }

        # Run the Google Search Scraper actor
        actor = client.actor("apify/google-search-scraper")
        run = actor.call(run_input=run_input, wait_secs=self.wait_for_finish)

        if not run:
            return []

        # Get dataset results
        dataset_id = run.get("defaultDatasetId")
        if dataset_id:
            dataset = client.dataset(dataset_id)
            items = list(dataset.iterate_items())
            return items

        return []


class ApifyInstagramScraper(BaseNode):
    """
    Scrape Instagram profiles, posts, comments, and hashtags.
    Extract user data, post details, engagement metrics, and more.
    apify, instagram, social, media, scraping, posts, profiles
    """

    usernames: List[str] = Field(
        default_factory=list,
        description="List of Instagram usernames to scrape"
    )
    hashtags: List[str] = Field(
        default_factory=list,
        description="List of hashtags to scrape"
    )
    results_limit: int = Field(
        default=50,
        description="Maximum number of posts to scrape per profile/hashtag"
    )
    scrape_comments: bool = Field(
        default=False,
        description="Whether to scrape comments on posts"
    )
    scrape_likes: bool = Field(
        default=False,
        description="Whether to scrape likes on posts"
    )
    wait_for_finish: int = Field(
        default=600,
        description="Maximum time to wait for scraping to complete (seconds)"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> List[Dict[str, Any]]:
        if not self.usernames and not self.hashtags:
            raise ValueError("Either usernames or hashtags is required")

        client, error = await _get_apify_client(context)
        if error:
            raise ValueError(error.get("error", "Failed to configure Apify client"))

        # Prepare input for Instagram Scraper
        run_input = {
            "resultsLimit": self.results_limit,
            "scrapeComments": self.scrape_comments,
            "scrapeLikes": self.scrape_likes,
        }

        # Add usernames if provided
        if self.usernames:
            run_input["usernames"] = self.usernames

        # Add hashtags if provided
        if self.hashtags:
            run_input["hashtags"] = self.hashtags

        # Run the Instagram Scraper actor
        actor = client.actor("apify/instagram-scraper")
        run = actor.call(run_input=run_input, wait_secs=self.wait_for_finish)

        if not run:
            return []

        # Get dataset results
        dataset_id = run.get("defaultDatasetId")
        if dataset_id:
            dataset = client.dataset(dataset_id)
            items = list(dataset.iterate_items())
            return items

        return []


class ApifyAmazonScraper(BaseNode):
    """
    Scrape Amazon product data including prices, reviews, and ratings.
    Extract product details, seller information, and customer reviews.
    apify, amazon, ecommerce, products, scraping, prices, reviews
    """

    search_queries: List[str] = Field(
        default_factory=list,
        description="List of search queries to execute on Amazon"
    )
    product_urls: List[str] = Field(
        default_factory=list,
        description="List of Amazon product URLs to scrape"
    )
    country_code: str = Field(
        default="US",
        description="Amazon country code (US, UK, DE, FR, ES, IT, etc.)"
    )
    max_items: int = Field(
        default=20,
        description="Maximum number of products to scrape per search"
    )
    scrape_reviews: bool = Field(
        default=False,
        description="Whether to scrape product reviews"
    )
    wait_for_finish: int = Field(
        default=600,
        description="Maximum time to wait for scraping to complete (seconds)"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> List[Dict[str, Any]]:
        if not self.search_queries and not self.product_urls:
            raise ValueError("Either search_queries or product_urls is required")

        client, error = await _get_apify_client(context)
        if error:
            raise ValueError(error.get("error", "Failed to configure Apify client"))

        # Prepare input for Amazon Scraper
        run_input = {
            "countryCode": self.country_code,
            "maxItems": self.max_items,
            "scrapeReviews": self.scrape_reviews,
        }

        # Add search queries if provided
        if self.search_queries:
            run_input["searchQueries"] = self.search_queries

        # Add product URLs if provided
        if self.product_urls:
            run_input["productUrls"] = self.product_urls

        # Run the Amazon Scraper actor
        actor = client.actor("apify/amazon-product-scraper")
        run = actor.call(run_input=run_input, wait_secs=self.wait_for_finish)

        if not run:
            return []

        # Get dataset results
        dataset_id = run.get("defaultDatasetId")
        if dataset_id:
            dataset = client.dataset(dataset_id)
            items = list(dataset.iterate_items())
            return items

        return []


class ApifyYouTubeScraper(BaseNode):
    """
    Scrape YouTube videos, channels, and playlists.
    Extract video metadata, comments, channel info, and statistics.
    apify, youtube, video, scraping, social, media, channels
    """

    search_queries: List[str] = Field(
        default_factory=list,
        description="List of search queries to execute on YouTube"
    )
    video_urls: List[str] = Field(
        default_factory=list,
        description="List of YouTube video URLs to scrape"
    )
    channel_urls: List[str] = Field(
        default_factory=list,
        description="List of YouTube channel URLs to scrape"
    )
    max_results: int = Field(
        default=50,
        description="Maximum number of videos to scrape"
    )
    scrape_comments: bool = Field(
        default=False,
        description="Whether to scrape video comments"
    )
    wait_for_finish: int = Field(
        default=600,
        description="Maximum time to wait for scraping to complete (seconds)"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> List[Dict[str, Any]]:
        if not self.search_queries and not self.video_urls and not self.channel_urls:
            raise ValueError("At least one of search_queries, video_urls, or channel_urls is required")

        client, error = await _get_apify_client(context)
        if error:
            raise ValueError(error.get("error", "Failed to configure Apify client"))

        # Prepare input for YouTube Scraper
        run_input = {
            "maxResults": self.max_results,
            "scrapeComments": self.scrape_comments,
        }

        # Build startUrls array based on inputs
        start_urls = []
        
        if self.search_queries:
            for query in self.search_queries:
                start_urls.append({"url": f"https://www.youtube.com/results?search_query={quote(query)}"})
        
        if self.video_urls:
            for url in self.video_urls:
                start_urls.append({"url": url})
        
        if self.channel_urls:
            for url in self.channel_urls:
                start_urls.append({"url": url})

        run_input["startUrls"] = start_urls

        # Run the YouTube Scraper actor
        actor = client.actor("apify/youtube-scraper")
        run = actor.call(run_input=run_input, wait_secs=self.wait_for_finish)

        if not run:
            return []

        # Get dataset results
        dataset_id = run.get("defaultDatasetId")
        if dataset_id:
            dataset = client.dataset(dataset_id)
            items = list(dataset.iterate_items())
            return items

        return []


class ApifyTwitterScraper(BaseNode):
    """
    Scrape Twitter/X posts, profiles, and followers.
    Extract tweets, user information, and engagement metrics.
    apify, twitter, x, social, media, scraping, tweets, posts
    """

    search_terms: List[str] = Field(
        default_factory=list,
        description="List of search terms to find tweets"
    )
    usernames: List[str] = Field(
        default_factory=list,
        description="List of Twitter usernames to scrape"
    )
    tweet_urls: List[str] = Field(
        default_factory=list,
        description="List of specific tweet URLs to scrape"
    )
    max_tweets: int = Field(
        default=100,
        description="Maximum number of tweets to scrape"
    )
    wait_for_finish: int = Field(
        default=600,
        description="Maximum time to wait for scraping to complete (seconds)"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> List[Dict[str, Any]]:
        if not self.search_terms and not self.usernames and not self.tweet_urls:
            raise ValueError("At least one of search_terms, usernames, or tweet_urls is required")

        client, error = await _get_apify_client(context)
        if error:
            raise ValueError(error.get("error", "Failed to configure Apify client"))

        # Prepare input for Twitter Scraper
        start_urls = []
        
        if self.search_terms:
            for term in self.search_terms:
                start_urls.append(f"https://twitter.com/search?q={quote(term)}")
        
        if self.usernames:
            for username in self.usernames:
                start_urls.append(f"https://twitter.com/{username}")
        
        if self.tweet_urls:
            start_urls.extend(self.tweet_urls)

        run_input = {
            "startUrls": start_urls,
            "maxItems": self.max_tweets,
        }

        # Run the Twitter Scraper actor
        actor = client.actor("apify/twitter-scraper")
        run = actor.call(run_input=run_input, wait_secs=self.wait_for_finish)

        if not run:
            return []

        # Get dataset results
        dataset_id = run.get("defaultDatasetId")
        if dataset_id:
            dataset = client.dataset(dataset_id)
            items = list(dataset.iterate_items())
            return items

        return []


class ApifyLinkedInScraper(BaseNode):
    """
    Scrape LinkedIn profiles, company pages, and job postings.
    Extract professional information, connections, and company data.
    apify, linkedin, professional, social, scraping, profiles, jobs
    """

    profile_urls: List[str] = Field(
        default_factory=list,
        description="List of LinkedIn profile URLs to scrape"
    )
    company_urls: List[str] = Field(
        default_factory=list,
        description="List of LinkedIn company page URLs to scrape"
    )
    job_search_urls: List[str] = Field(
        default_factory=list,
        description="List of LinkedIn job search URLs"
    )
    max_results: int = Field(
        default=50,
        description="Maximum number of results to scrape"
    )
    wait_for_finish: int = Field(
        default=600,
        description="Maximum time to wait for scraping to complete (seconds)"
    )

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> List[Dict[str, Any]]:
        if not self.profile_urls and not self.company_urls and not self.job_search_urls:
            raise ValueError("At least one of profile_urls, company_urls, or job_search_urls is required")

        client, error = await _get_apify_client(context)
        if error:
            raise ValueError(error.get("error", "Failed to configure Apify client"))

        # Prepare input for LinkedIn Scraper
        start_urls = []
        start_urls.extend(self.profile_urls)
        start_urls.extend(self.company_urls)
        start_urls.extend(self.job_search_urls)

        run_input = {
            "startUrls": [{"url": url} for url in start_urls],
            "maxResults": self.max_results,
        }

        # Run the LinkedIn Scraper actor
        actor = client.actor("apify/linkedin-profile-scraper")
        run = actor.call(run_input=run_input, wait_secs=self.wait_for_finish)

        if not run:
            return []

        # Get dataset results
        dataset_id = run.get("defaultDatasetId")
        if dataset_id:
            dataset = client.dataset(dataset_id)
            items = list(dataset.iterate_items())
            return items

        return []
