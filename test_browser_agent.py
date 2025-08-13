import asyncio
from nodetool.nodes.lib.browser import Browser
from nodetool.workflows.processing_context import ProcessingContext


async def main():
    """
    Simple entrypoint to extract content from a well-known page.
    Uses WebFetch to download and convert the page to text.
    """
    context = ProcessingContext()
    browser = Browser(url="https://en.wikipedia.org/wiki/Test-driven_development")
    content = await browser.process(context)
    print(content)


if __name__ == "__main__":
    asyncio.run(main())
