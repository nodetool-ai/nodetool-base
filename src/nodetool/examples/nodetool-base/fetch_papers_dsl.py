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

import os
from nodetool.dsl.graph import graph_result, run_graph, graph
from nodetool.dsl.lib.http import GetRequest, DownloadFiles
from nodetool.dsl.lib.markdown import ExtractLinks
from nodetool.dsl.nodetool.data import FromList, Filter, ExtractColumn
from nodetool.dsl.nodetool.output import ListOutput


async def example():
    """
    Full workflow including downloading papers.
    """
    # Fetch README
    fetch_readme = GetRequest(
        url="https://raw.githubusercontent.com/abacaj/awesome-transformers/refs/heads/main/README.md",
    )

    # Extract links
    extract_links = ExtractLinks(
        markdown=fetch_readme.output,
        include_titles=True,
    )

    # Convert to DataFrame
    links_df = FromList(values=extract_links.output)

    # Filter papers
    filtered_df = Filter(
        df=links_df.output,
        condition="title == 'Paper'",
    )

    # Extract URLs
    urls = ExtractColumn(
        dataframe=filtered_df.output,
        column_name="url",
    )

    # Download papers
    downloader = DownloadFiles(
        urls=urls.output,
        output_folder="papers",
        max_concurrent_downloads=5,
    )

    # Create and run the graph
    g = graph(downloader)
    result = await run_graph(g)
    return result


if __name__ == "__main__":
    import asyncio

    # Just show the preview
    urls = asyncio.run(example())

    for file in os.listdir("papers"):
        print(f"Downloaded paper: {file}")

