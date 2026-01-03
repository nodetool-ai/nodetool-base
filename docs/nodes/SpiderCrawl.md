# SpiderCrawl Node

The SpiderCrawl node is a powerful web crawler that discovers and extracts content from websites following links in a breadth-first manner. It's designed for agentic workflows and follows web spider best practices.

## Features

### Spider Best Practices
- **Robots.txt compliance**: Respects website crawling rules (configurable)
- **Rate limiting**: Configurable delays between requests (default 1 second)
- **Depth control**: Prevents infinite crawling with max depth limits
- **Safety limits**: Maximum page count to prevent runaway crawls
- **Politeness**: User-agent identification and respectful crawling

### Flexible URL Filtering
- **Domain restriction**: Option to stay within the same domain
- **Pattern matching**: Include only URLs matching a regex pattern
- **Exclusion patterns**: Exclude URLs matching a regex pattern (e.g., skip images/CSS)
- **Automatic deduplication**: Tracks visited URLs to avoid duplicates
- **Fragment removal**: Normalizes URLs by removing fragments

### Streaming Output
- **Async generator**: Emits URLs as they are discovered using `gen_process`
- **Real-time processing**: Process pages as they're crawled, not after completion
- **Memory efficient**: Doesn't load all pages into memory at once
- **Progress tracking**: Monitor crawl progress in real-time

### Agentic Workflow Optimization
- **Optional HTML content**: Choose whether to include page HTML
- **Rich metadata**: URL, title, status code, and depth for each page
- **Error resilience**: Continues crawling even if some pages fail
- **Timeout control**: Configurable timeouts prevent hanging on slow pages

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_url` | string | required | Starting URL to begin crawling |
| `max_depth` | int | 2 | Maximum depth to crawl (0 = start page only) |
| `max_pages` | int | 50 | Maximum number of pages to crawl |
| `same_domain_only` | bool | true | Only follow links within the same domain |
| `include_html` | bool | false | Include HTML content in output |
| `respect_robots_txt` | bool | true | Respect robots.txt rules |
| `delay_ms` | int | 1000 | Delay in milliseconds between requests |
| `timeout` | int | 30000 | Timeout in milliseconds for each page |
| `url_pattern` | string | "" | Optional regex pattern to filter URLs |
| `exclude_pattern` | string | "" | Optional regex to exclude URLs |

## Output Format

Each discovered page yields the following structure:

```python
{
    "url": str,           # The discovered URL
    "depth": int,         # Depth level from start (0 = start page)
    "html": str | None,   # HTML content (if include_html=True)
    "title": str | None,  # Page title
    "status_code": int    # HTTP status code (200 = success)
}
```

## Usage Examples

### Basic Site Crawl
```python
from nodetool.dsl.lib.browser import SpiderCrawl

spider = SpiderCrawl(
    start_url="https://example.com",
    max_depth=2,
    max_pages=50
)
```

### Focused Crawl with Filtering
```python
# Only crawl blog posts
spider = SpiderCrawl(
    start_url="https://example.com",
    max_depth=3,
    url_pattern=r"/blog/.*",
    exclude_pattern=r"\.(jpg|png|gif|pdf)$",
    delay_ms=2000  # Be extra polite
)
```

### Deep Content Extraction
```python
# Crawl with HTML content for analysis
spider = SpiderCrawl(
    start_url="https://docs.example.com",
    max_depth=4,
    max_pages=200,
    include_html=True,  # Include HTML for processing
    same_domain_only=True
)
```

## Best Practices

### Performance
- Set `include_html=False` unless you need the content
- Use appropriate `delay_ms` values (1000ms minimum for external sites)
- Set reasonable `max_pages` limits to prevent long-running crawls
- Use `url_pattern` and `exclude_pattern` to focus the crawl

### Politeness
- Always respect `robots.txt` for external sites
- Use delays of 1-2 seconds between requests
- Identify your crawler with a descriptive user agent
- Set appropriate timeout values

### Reliability
- Handle errors gracefully (spider continues on page errors)
- Monitor progress using the depth field
- Set timeout values appropriate for your target sites
- Use `max_depth` to prevent infinite recursion

## Common Patterns

### Documentation Crawler
```python
SpiderCrawl(
    start_url="https://docs.site.com",
    max_depth=4,
    url_pattern=r"/docs/",
    include_html=True,
    max_pages=200
)
```

### Blog Discovery
```python
SpiderCrawl(
    start_url="https://blog.site.com",
    max_depth=2,
    url_pattern=r"/\d{4}/",
    exclude_pattern=r"\.(jpg|png|pdf)$"
)
```

### Link Validation
```python
SpiderCrawl(
    start_url="https://mysite.com",
    max_depth=3,
    include_html=False,
    max_pages=500
)
# Check status_code for each page
```
