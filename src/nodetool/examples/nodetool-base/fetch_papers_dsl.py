"""
Fetch Papers DSL Example

Automatically fetch and download research papers from the Awesome Transformers GitHub repository.

Workflow:
1. **HTTP GET Request** - Fetch the README.md from GitHub
2. **Extract Links** - Extract all markdown links with titles
3. **Convert to DataFrame** - Create a structured DataFrame from the links
4. **Filter** - Keep only entries with title "Paper"
5. **Extract Column** - Get the URL column from filtered results
6. **Preview** - Display the paper URLs
7. **Download** - Download all papers to a folder
"""

from nodetool.dsl.graph import graph_result, run_graph, graph
from nodetool.dsl.lib.http import GetRequest, DownloadFiles
from nodetool.dsl.lib.markdown import ExtractLinks
from nodetool.dsl.nodetool.data import FromList, Filter, ExtractColumn
from nodetool.dsl.nodetool.output import ListOutput


async def example():
    """
    Fetch research papers from awesome-transformers repository.
    """
    # Fetch the README from GitHub
    fetch_readme = GetRequest(
        url="https://raw.githubusercontent.com/abacaj/awesome-transformers/refs/heads/main/README.md",
        auth=None,
    )

    # Extract links from markdown
    extract_links = ExtractLinks(
        markdown=fetch_readme,
        include_titles=True,
    )

    # Convert links to DataFrame
    links_df = FromList(values=extract_links)

    # Filter to keep only "Paper" entries
    filtered_df = Filter(
        df=links_df,
        condition="title == 'Paper'",
    )

    # Extract URL column
    urls = ExtractColumn(
        df=filtered_df,
        column_name="url",
    )

    # Output the URLs
    output = ListOutput(
        name="paper_urls",
        value=urls,
    )

    result = await graph_result(output)
    return result


async def example_with_download():
    """
    Full workflow including downloading papers.
    """
    # Fetch README
    fetch_readme = GetRequest(
        url="https://raw.githubusercontent.com/abacaj/awesome-transformers/refs/heads/main/README.md",
        auth=None,
    )

    # Extract links
    extract_links = ExtractLinks(
        markdown=fetch_readme,
        include_titles=True,
    )

    # Convert to DataFrame
    links_df = FromList(values=extract_links)

    # Filter papers
    filtered_df = Filter(
        df=links_df,
        condition="title == 'Paper'",
    )

    # Extract URLs
    urls = ExtractColumn(
        df=filtered_df,
        column_name="url",
    )

    # Download papers
    downloader = DownloadFiles(
        urls=urls,
        output_folder={},
        auth=None,
        max_concurrent_downloads=5,
    )

    # Create and run the graph
    g = graph(downloader)
    result = await run_graph(g)
    return result


async def example_with_preview():
    """
    Preview papers before downloading.
    """
    fetch_readme = GetRequest(
        url="https://raw.githubusercontent.com/abacaj/awesome-transformers/refs/heads/main/README.md",
        auth=None,
    )

    extract_links = ExtractLinks(
        markdown=fetch_readme,
        include_titles=True,
    )

    links_df = FromList(values=extract_links)

    filtered_df = Filter(
        df=links_df,
        condition="title == 'Paper'",
    )

    urls = ExtractColumn(
        df=filtered_df,
        column_name="url",
    )

    # Output URLs
    output_urls = ListOutput(
        name="paper_urls",
        value=urls,
    )

    # Then download
    downloader = DownloadFiles(
        urls=urls,
        output_folder={},
        auth=None,
        max_concurrent_downloads=5,
    )

    g = graph(output_urls, downloader)
    result = await run_graph(g)
    return result


if __name__ == "__main__":
    import asyncio

    # Just show the preview
    urls = asyncio.run(example())
    print(f"Found {len(urls) if isinstance(urls, list) else 'some'} papers")

    # Or download
    # result = asyncio.run(example_with_download())
    # print(f"Download complete: {result}")
