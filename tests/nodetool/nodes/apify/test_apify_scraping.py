import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.apify.scraping import (
    ApifyWebScraper,
    ApifyGoogleSearchScraper,
    ApifyInstagramScraper,
    ApifyAmazonScraper,
    ApifyYouTubeScraper,
    ApifyTwitterScraper,
    ApifyLinkedInScraper,
)


@pytest.fixture
def context():
    """Create a test processing context."""
    ctx = ProcessingContext(user_id="test", auth_token="test")
    return ctx


@pytest.fixture
def mock_apify_client():
    """Mock Apify client for testing."""
    with patch('nodetool.nodes.apify.scraping.ApifyClient') as mock_client:
        yield mock_client


def test_apify_web_scraper_instantiation():
    """Test that ApifyWebScraper can be instantiated."""
    node = ApifyWebScraper(
        start_urls=["https://example.com"],
        max_pages=5,
    )
    assert node.start_urls == ["https://example.com"]
    assert node.max_pages == 5
    assert node.link_selector == "a[href]"


def test_apify_web_scraper_default_values():
    """Test ApifyWebScraper default values."""
    node = ApifyWebScraper()
    assert node.start_urls == []
    assert node.link_selector == "a[href]"
    assert node.max_pages == 10
    assert node.wait_for_finish == 300


@pytest.mark.asyncio
async def test_apify_web_scraper_missing_start_urls(context: ProcessingContext):
    """Test ApifyWebScraper raises error without start_urls."""
    node = ApifyWebScraper(start_urls=[])
    with pytest.raises(ValueError, match="start_urls is required"):
        await node.process(context)


@pytest.mark.asyncio
async def test_apify_web_scraper_missing_api_token(context: ProcessingContext):
    """Test ApifyWebScraper raises error without API token."""
    node = ApifyWebScraper(start_urls=["https://example.com"])
    with pytest.raises(ValueError, match="APIFY_API_TOKEN not configured"):
        await node.process(context)


@pytest.mark.asyncio
async def test_apify_web_scraper_process(context: ProcessingContext, mock_apify_client):
    """Test ApifyWebScraper process method."""
    # Mock the environment to return API token
    context.get_env = MagicMock(return_value="test_token")
    
    # Mock the client and actor
    mock_client_instance = MagicMock()
    mock_apify_client.return_value = mock_client_instance
    
    mock_actor = MagicMock()
    mock_client_instance.actor.return_value = mock_actor
    
    # Mock the run result
    mock_run = {"defaultDatasetId": "test_dataset_id"}
    mock_actor.call.return_value = mock_run
    
    # Mock the dataset
    mock_dataset = MagicMock()
    mock_client_instance.dataset.return_value = mock_dataset
    mock_dataset.iterate_items.return_value = [
        {"url": "https://example.com", "title": "Example"},
    ]
    
    node = ApifyWebScraper(start_urls=["https://example.com"])
    result = await node.process(context)
    
    assert len(result) == 1
    assert result[0]["url"] == "https://example.com"


def test_apify_google_search_scraper_instantiation():
    """Test that ApifyGoogleSearchScraper can be instantiated."""
    node = ApifyGoogleSearchScraper(
        queries=["test query"],
        country_code="us",
        max_pages=2,
    )
    assert node.queries == ["test query"]
    assert node.country_code == "us"
    assert node.max_pages == 2


def test_apify_google_search_scraper_default_values():
    """Test ApifyGoogleSearchScraper default values."""
    node = ApifyGoogleSearchScraper()
    assert node.queries == []
    assert node.country_code == "us"
    assert node.language_code == "en"
    assert node.max_pages == 1
    assert node.results_per_page == 100


@pytest.mark.asyncio
async def test_apify_google_search_scraper_missing_queries(context: ProcessingContext):
    """Test ApifyGoogleSearchScraper raises error without queries."""
    node = ApifyGoogleSearchScraper(queries=[])
    with pytest.raises(ValueError, match="queries is required"):
        await node.process(context)


def test_apify_instagram_scraper_instantiation():
    """Test that ApifyInstagramScraper can be instantiated."""
    node = ApifyInstagramScraper(
        usernames=["testuser"],
        results_limit=100,
    )
    assert node.usernames == ["testuser"]
    assert node.results_limit == 100


def test_apify_instagram_scraper_default_values():
    """Test ApifyInstagramScraper default values."""
    node = ApifyInstagramScraper()
    assert node.usernames == []
    assert node.hashtags == []
    assert node.results_limit == 50
    assert node.scrape_comments is False
    assert node.scrape_likes is False


@pytest.mark.asyncio
async def test_apify_instagram_scraper_missing_inputs(context: ProcessingContext):
    """Test ApifyInstagramScraper raises error without usernames or hashtags."""
    node = ApifyInstagramScraper()
    with pytest.raises(ValueError, match="Either usernames or hashtags is required"):
        await node.process(context)


def test_apify_amazon_scraper_instantiation():
    """Test that ApifyAmazonScraper can be instantiated."""
    node = ApifyAmazonScraper(
        search_queries=["laptop"],
        country_code="US",
        max_items=30,
    )
    assert node.search_queries == ["laptop"]
    assert node.country_code == "US"
    assert node.max_items == 30


def test_apify_amazon_scraper_default_values():
    """Test ApifyAmazonScraper default values."""
    node = ApifyAmazonScraper()
    assert node.search_queries == []
    assert node.product_urls == []
    assert node.country_code == "US"
    assert node.max_items == 20
    assert node.scrape_reviews is False


@pytest.mark.asyncio
async def test_apify_amazon_scraper_missing_inputs(context: ProcessingContext):
    """Test ApifyAmazonScraper raises error without search_queries or product_urls."""
    node = ApifyAmazonScraper()
    with pytest.raises(ValueError, match="Either search_queries or product_urls is required"):
        await node.process(context)


def test_apify_youtube_scraper_instantiation():
    """Test that ApifyYouTubeScraper can be instantiated."""
    node = ApifyYouTubeScraper(
        search_queries=["python tutorial"],
        max_results=100,
    )
    assert node.search_queries == ["python tutorial"]
    assert node.max_results == 100


def test_apify_youtube_scraper_default_values():
    """Test ApifyYouTubeScraper default values."""
    node = ApifyYouTubeScraper()
    assert node.search_queries == []
    assert node.video_urls == []
    assert node.channel_urls == []
    assert node.max_results == 50
    assert node.scrape_comments is False


@pytest.mark.asyncio
async def test_apify_youtube_scraper_missing_inputs(context: ProcessingContext):
    """Test ApifyYouTubeScraper raises error without any input URLs."""
    node = ApifyYouTubeScraper()
    with pytest.raises(ValueError, match="At least one of"):
        await node.process(context)


def test_apify_twitter_scraper_instantiation():
    """Test that ApifyTwitterScraper can be instantiated."""
    node = ApifyTwitterScraper(
        search_terms=["python"],
        max_tweets=200,
    )
    assert node.search_terms == ["python"]
    assert node.max_tweets == 200


def test_apify_twitter_scraper_default_values():
    """Test ApifyTwitterScraper default values."""
    node = ApifyTwitterScraper()
    assert node.search_terms == []
    assert node.usernames == []
    assert node.tweet_urls == []
    assert node.max_tweets == 100


@pytest.mark.asyncio
async def test_apify_twitter_scraper_missing_inputs(context: ProcessingContext):
    """Test ApifyTwitterScraper raises error without any inputs."""
    node = ApifyTwitterScraper()
    with pytest.raises(ValueError, match="At least one of"):
        await node.process(context)


def test_apify_linkedin_scraper_instantiation():
    """Test that ApifyLinkedInScraper can be instantiated."""
    node = ApifyLinkedInScraper(
        profile_urls=["https://linkedin.com/in/test"],
        max_results=50,
    )
    assert node.profile_urls == ["https://linkedin.com/in/test"]
    assert node.max_results == 50


def test_apify_linkedin_scraper_default_values():
    """Test ApifyLinkedInScraper default values."""
    node = ApifyLinkedInScraper()
    assert node.profile_urls == []
    assert node.company_urls == []
    assert node.job_search_urls == []
    assert node.max_results == 50


@pytest.mark.asyncio
async def test_apify_linkedin_scraper_missing_inputs(context: ProcessingContext):
    """Test ApifyLinkedInScraper raises error without any inputs."""
    node = ApifyLinkedInScraper()
    with pytest.raises(ValueError, match="At least one of"):
        await node.process(context)
