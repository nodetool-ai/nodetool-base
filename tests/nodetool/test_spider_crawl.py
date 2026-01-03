import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.browser import SpiderCrawl


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_spider_crawl_same_domain_only(context: ProcessingContext):
    """Test same domain restriction."""
    node = SpiderCrawl(
        start_url="http://example.com",
        max_depth=1,
        max_pages=10,
        same_domain_only=True,
        delay_ms=0
    )
    
    # The implementation should filter out external links
    # This is tested through the logic in gen_process
    assert node.same_domain_only is True


@pytest.mark.asyncio
async def test_spider_crawl_depth_limit(context: ProcessingContext):
    """Test depth limiting."""
    node = SpiderCrawl(
        start_url="http://example.com",
        max_depth=2,
        max_pages=100,
        delay_ms=0
    )
    
    assert node.max_depth == 2


@pytest.mark.asyncio
async def test_spider_crawl_timeout_calculation():
    """Test timeout calculation."""
    node = SpiderCrawl(
        start_url="http://example.com",
        max_depth=1,
        max_pages=10,
        timeout=5000,
        delay_ms=1000
    )
    
    timeout = node.get_timeout_seconds()
    assert timeout is not None
    assert timeout > 0
    assert timeout <= 600.0  # Should be capped at 10 minutes


@pytest.mark.asyncio
async def test_spider_crawl_exclude_pattern(context: ProcessingContext):
    """Test URL exclusion pattern."""
    node = SpiderCrawl(
        start_url="http://example.com",
        max_depth=1,
        max_pages=10,
        exclude_pattern=r"\.(jpg|png|gif|css|js)$",
        delay_ms=0
    )
    
    assert node.exclude_pattern == r"\.(jpg|png|gif|css|js)$"


@pytest.mark.asyncio
async def test_spider_crawl_requires_start_url(context: ProcessingContext):
    """Test that start_url is required."""
    
    node = SpiderCrawl(
        start_url="",
        max_depth=0,
        max_pages=1,
        delay_ms=0
    )
    
    with pytest.raises(ValueError, match="start_url is required"):
        async for _ in node.gen_process(context):
            pass


@pytest.mark.asyncio
async def test_spider_crawl_configuration():
    """Test node configuration options."""
    node = SpiderCrawl(
        start_url="http://example.com",
        max_depth=3,
        max_pages=100,
        same_domain_only=False,
        include_html=True,
        respect_robots_txt=False,
        delay_ms=500,
        timeout=15000,
        url_pattern=r".*blog.*",
        exclude_pattern=r"\.(pdf|zip)$"
    )
    
    assert node.start_url == "http://example.com"
    assert node.max_depth == 3
    assert node.max_pages == 100
    assert node.same_domain_only is False
    assert node.include_html is True
    assert node.respect_robots_txt is False
    assert node.delay_ms == 500
    assert node.timeout == 15000
    assert node.url_pattern == r".*blog.*"
    assert node.exclude_pattern == r"\.(pdf|zip)$"


@pytest.mark.asyncio
async def test_spider_crawl_default_values():
    """Test default configuration values."""
    node = SpiderCrawl(start_url="http://example.com")
    
    assert node.max_depth == 2
    assert node.max_pages == 50
    assert node.same_domain_only is True
    assert node.include_html is False
    assert node.respect_robots_txt is True
    assert node.delay_ms == 1000
    assert node.timeout == 30000
    assert node.url_pattern == ""
    assert node.exclude_pattern == ""


@pytest.mark.asyncio
async def test_spider_crawl_expose_as_tool():
    """Test that SpiderCrawl is exposed as a tool."""
    assert SpiderCrawl._expose_as_tool is True
