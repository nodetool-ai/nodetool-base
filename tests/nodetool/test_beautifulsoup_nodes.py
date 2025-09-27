import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.beautifulsoup import (
    BaseUrl,
    ExtractLinks,
    ExtractMetadata,
    ExtractImages,
    ExtractVideos,
    ExtractAudio,
    WebsiteContentExtractor,
    HTMLToText,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


HTML_SAMPLE = """
<!doctype html>
<html>
  <head>
    <title>Sample Page</title>
    <meta name="description" content="Example description" />
    <meta name="keywords" content="alpha, beta" />
  </head>
  <body>
    <nav>nav-links</nav>
    <article>
      <h1>Heading</h1>
      <p>Para 1</p>
      <a href="/internal">Internal</a>
      <a href="https://ext.example/x">External</a>
      <img src="/img.png" />
      <video src="/v.mp4"></video>
      <iframe src="/embed.mp4"></iframe>
      <audio src="/song.mp3"></audio>
      <source src="/alt.ogg" />
    </article>
    <footer>footer</footer>
  </body>
</html>
    """


@pytest.mark.asyncio
async def test_base_url(context: ProcessingContext):
    node = BaseUrl(url="https://example.com/path?q=1")
    assert await node.process(context) == "https://example.com"


@pytest.mark.asyncio
async def test_extract_links(context: ProcessingContext):
    node = ExtractLinks(html=HTML_SAMPLE, base_url="https://example.com")
    df = await node.process(context)
    # Ensure rows contain href, text, type and classify internal/external
    rows = df.data
    hrefs = [r[0] for r in rows]
    types = {r[2] for r in rows}
    assert "/internal" in hrefs and "https://ext.example/x" in hrefs
    assert types == {"internal", "external"}


@pytest.mark.asyncio
async def test_extract_metadata(context: ProcessingContext):
    node = ExtractMetadata(html=HTML_SAMPLE)
    meta = await node.process(context)
    assert meta["title"] == "Sample Page"
    assert meta["description"] == "Example description"
    assert meta["keywords"] == "alpha, beta"


@pytest.mark.asyncio
async def test_extract_images_videos_audio(context: ProcessingContext):
    base = "https://example.com/base/"
    imgs = []
    vids = []
    auds = []
    async for item in ExtractImages(html=HTML_SAMPLE, base_url=base).gen_process(
        context
    ):
        imgs.append(item["image"])

    async for item in ExtractVideos(html=HTML_SAMPLE, base_url=base).gen_process(
        context
    ):
        vids.append(item["video"])

    async for item in ExtractAudio(html=HTML_SAMPLE, base_url=base).gen_process(
        context
    ):
        auds.append(item["audio"])

    assert any(img.uri.endswith("/img.png") for img in imgs)
    assert any(v.uri.endswith("/v.mp4") for v in vids) and any(
        v.uri.endswith("/embed.mp4") for v in vids
    )
    assert any(a.uri.endswith("/song.mp3") for a in auds) and any(
        a.uri.endswith("/alt.ogg") for a in auds
    )


@pytest.mark.asyncio
async def test_website_content_extractor(context: ProcessingContext):
    text = await WebsiteContentExtractor(html_content=HTML_SAMPLE).process(context)
    assert "Heading" in text and "Para 1" in text
    # Should strip nav/footer text
    assert "nav-links" not in text and "footer" not in text


@pytest.mark.asyncio
async def test_html_to_text(context: ProcessingContext):
    html = "<p>Hello<br/>World</p>"
    node = HTMLToText(text=html, preserve_linebreaks=True)
    out = await node.process(context)
    assert "Hello" in out and "World" in out
