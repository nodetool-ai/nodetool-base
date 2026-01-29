"""
Website Spider Example – DSL workflow

Demonstrates how to use the SpiderCrawl node to discover and process
web pages from a starting URL. This example crawls a website and 
collects URLs, titles, and page content for further processing.
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.lib.browser import SpiderCrawl
from nodetool.dsl.nodetool.output import Output

# Configure the starting URL for the spider
start_url = StringInput(
    name="start_url",
    description="Website URL to start crawling",
    value="https://example.com",
)

# Configure spider crawl with sensible defaults for agentic workflows
spider = SpiderCrawl(
    start_url=start_url.output,
    max_depth=2,  # Crawl start page and linked pages
    max_pages=20,  # Limit to 20 pages for demonstration
    same_domain_only=True,  # Only follow links on the same domain
    include_html=False,  # Don't include HTML content (faster, less bandwidth)
    respect_robots_txt=True,  # Respect robots.txt rules
    delay_ms=1000,  # 1 second delay between requests (polite crawling)
    timeout=30000,  # 30 second timeout per page
)

# Output the discovered URLs
output = Output(
    name="discovered_urls",
    value=spider.output,
)

graph = create_graph(output)

if __name__ == "__main__":
    """
    Run the spider crawl workflow.
    
    The spider will:
    1. Start at the provided URL
    2. Extract all links from each page
    3. Follow links up to max_depth levels
    4. Filter URLs based on same_domain_only and pattern settings
    5. Emit each discovered URL with metadata (title, status code, depth)
    
    Use cases:
    - Build a sitemap of a website
    - Collect URLs for bulk content analysis
    - Discover all pages in a documentation site
    - Feed URLs to an agentic workflow for processing
    - Find specific pages matching a pattern
    """
    results = run_graph(graph)
    
    print("Spider Crawl Results:")
    print(f"Total URLs discovered: {len(results['discovered_urls'])}")
    print("\nDiscovered pages:")
    
    for page in results['discovered_urls']:
        print(f"  Depth {page['depth']}: {page['url']}")
        print(f"    Title: {page['title']}")
        print(f"    Status: {page['status_code']}")
        print()
