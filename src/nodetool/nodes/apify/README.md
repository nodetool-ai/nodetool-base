# Apify Nodes

This package provides integration with [Apify](https://apify.com/) web scraping and automation actors.

## Overview

Apify is a web scraping and automation platform that provides pre-built "actors" for extracting data from websites, social media platforms, and e-commerce sites. These nodes allow you to use popular Apify actors directly in your nodetool workflows.

## Setup

To use Apify nodes, you need an Apify API token:

1. Sign up for a free account at [apify.com](https://apify.com/)
2. Get your API token from the [Apify Console](https://console.apify.com/account/integrations)
3. Set the `APIFY_API_TOKEN` environment variable in your nodetool configuration

## Available Nodes

### ApifyWebScraper

General-purpose web scraper using CSS selectors and custom JavaScript.

**Inputs:**
- `start_urls` (List[str]): URLs to scrape
- `link_selector` (str): CSS selector for links to follow (default: "a[href]")
- `page_function` (str): JavaScript function to execute on each page
- `max_pages` (int): Maximum number of pages to scrape (default: 10)
- `wait_for_finish` (int): Maximum time to wait in seconds (default: 300)

**Output:** List of dictionaries containing scraped data

**Example use cases:**
- Extract product information from e-commerce sites
- Scrape article content from blogs and news sites
- Collect structured data from directory listings

---

### ApifyGoogleSearchScraper

Scrape Google Search results including organic results, ads, and related searches.

**Inputs:**
- `queries` (List[str]): Search queries to execute
- `country_code` (str): Country code (default: "us")
- `language_code` (str): Language code (default: "en")
- `max_pages` (int): Maximum result pages per query (default: 1)
- `results_per_page` (int): Results per page, 10-100 (default: 100)
- `wait_for_finish` (int): Maximum time to wait in seconds (default: 300)

**Output:** List of dictionaries containing search results with titles, URLs, descriptions, and positions

**Example use cases:**
- SEO analysis and keyword research
- Competitor monitoring
- Market research

---

### ApifyInstagramScraper

Scrape Instagram profiles, posts, comments, and hashtags.

**Inputs:**
- `usernames` (List[str]): Instagram usernames to scrape
- `hashtags` (List[str]): Hashtags to scrape
- `results_limit` (int): Maximum posts per profile/hashtag (default: 50)
- `scrape_comments` (bool): Whether to scrape comments (default: False)
- `scrape_likes` (bool): Whether to scrape likes (default: False)
- `wait_for_finish` (int): Maximum time to wait in seconds (default: 600)

**Output:** List of dictionaries containing post data, engagement metrics, and user information

**Example use cases:**
- Social media monitoring
- Influencer analysis
- Brand tracking

---

### ApifyAmazonScraper

Scrape Amazon product data including prices, reviews, and ratings.

**Inputs:**
- `search_queries` (List[str]): Search queries to execute on Amazon
- `product_urls` (List[str]): Specific product URLs to scrape
- `country_code` (str): Amazon country code (default: "US")
- `max_items` (int): Maximum products to scrape (default: 20)
- `scrape_reviews` (bool): Whether to scrape reviews (default: False)
- `wait_for_finish` (int): Maximum time to wait in seconds (default: 600)

**Output:** List of dictionaries containing product information, prices, ratings, and seller details

**Example use cases:**
- Price monitoring and comparison
- Product research
- Market analysis

---

### ApifyYouTubeScraper

Scrape YouTube videos, channels, and playlists.

**Inputs:**
- `search_queries` (List[str]): Search queries for YouTube
- `video_urls` (List[str]): Specific video URLs to scrape
- `channel_urls` (List[str]): Channel URLs to scrape
- `max_results` (int): Maximum videos to scrape (default: 50)
- `scrape_comments` (bool): Whether to scrape comments (default: False)
- `wait_for_finish` (int): Maximum time to wait in seconds (default: 600)

**Output:** List of dictionaries containing video metadata, view counts, likes, and channel information

**Example use cases:**
- Content analysis
- Competitor research
- Trend monitoring

---

### ApifyTwitterScraper

Scrape Twitter/X posts, profiles, and followers.

**Inputs:**
- `search_terms` (List[str]): Search terms to find tweets
- `usernames` (List[str]): Twitter usernames to scrape
- `tweet_urls` (List[str]): Specific tweet URLs to scrape
- `max_tweets` (int): Maximum tweets to scrape (default: 100)
- `wait_for_finish` (int): Maximum time to wait in seconds (default: 600)

**Output:** List of dictionaries containing tweets, engagement metrics, and user information

**Example use cases:**
- Social media monitoring
- Sentiment analysis
- Trend tracking

---

### ApifyLinkedInScraper

Scrape LinkedIn profiles, company pages, and job postings.

**Inputs:**
- `profile_urls` (List[str]): LinkedIn profile URLs to scrape
- `company_urls` (List[str]): Company page URLs to scrape
- `job_search_urls` (List[str]): Job search URLs
- `max_results` (int): Maximum results to scrape (default: 50)
- `wait_for_finish` (int): Maximum time to wait in seconds (default: 600)

**Output:** List of dictionaries containing professional information, company data, or job details

**Example use cases:**
- Recruitment and talent sourcing
- Company research
- Market intelligence

---

## Important Notes

### Rate Limits and Costs

- Apify actors run on Apify's infrastructure and consume Apify compute units
- Free tier includes limited compute units per month
- Check the [Apify pricing page](https://apify.com/pricing) for current rates
- Each actor has different performance characteristics and costs

### Wait Times

- All nodes include a `wait_for_finish` parameter that controls how long to wait for the actor to complete
- Default wait times vary by node (300-600 seconds)
- Large scraping jobs may require longer wait times
- Jobs that exceed the wait time may be incomplete

### Data Quality

- The quality and structure of returned data depends on the specific Apify actor
- Some actors may return different data structures based on the input
- Always validate and handle the returned data appropriately in your workflow

### Legal and Ethical Considerations

- Respect website terms of service and robots.txt
- Follow rate limiting and scraping best practices
- Ensure compliance with data protection regulations (GDPR, CCPA, etc.)
- Some websites may block scraping attempts

## Troubleshooting

### "APIFY_API_TOKEN not configured" Error

Make sure the `APIFY_API_TOKEN` environment variable is set in your nodetool configuration.

### Timeout Errors

If actors are timing out:
- Increase the `wait_for_finish` parameter
- Reduce the amount of data being scraped (e.g., lower `max_results`)
- Check Apify Console for actor run status

### Empty Results

If you're getting empty results:
- Verify the input parameters are correct
- Check the Apify Console for error messages in the actor run logs
- Some actors may have changed their input format - refer to actor documentation

## References

- [Apify Documentation](https://docs.apify.com/)
- [Apify Store](https://apify.com/store)
- [Apify Python Client](https://docs.apify.com/api/client/python)
