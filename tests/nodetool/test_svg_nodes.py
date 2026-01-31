import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.svg import (
    RectNode,
    CircleNode,
    EllipseNode,
    LineNode,
    PolygonNode,
    PathNode,
    Text,
    SVGTextAnchor,
    GaussianBlur,
    DropShadow,
    Document,
    Gradient,
    Transform,
    ClipPath,
)
from nodetool.metadata.types import ColorRef, SVGElement


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


class TestRectNode:
    """Tests for RectNode SVG shape."""

    @pytest.mark.asyncio
    async def test_default_rect(self, context: ProcessingContext):
        node = RectNode()
        result = await node.process(context)
        assert result.name == "rect"
        assert result.attributes["x"] == "0"
        assert result.attributes["y"] == "0"
        assert result.attributes["width"] == "100"
        assert result.attributes["height"] == "100"

    @pytest.mark.asyncio
    async def test_custom_rect(self, context: ProcessingContext):
        node = RectNode(
            x=10,
            y=20,
            width=200,
            height=150,
            fill=ColorRef(value="#ff0000"),
            stroke=ColorRef(value="#0000ff"),
            stroke_width=3,
        )
        result = await node.process(context)
        assert result.attributes["x"] == "10"
        assert result.attributes["y"] == "20"
        assert result.attributes["width"] == "200"
        assert result.attributes["height"] == "150"
        assert result.attributes["stroke-width"] == "3"


class TestCircleNode:
    """Tests for CircleNode SVG shape."""

    @pytest.mark.asyncio
    async def test_default_circle(self, context: ProcessingContext):
        node = CircleNode()
        result = await node.process(context)
        assert result.name == "circle"
        assert result.attributes["cx"] == "0"
        assert result.attributes["cy"] == "0"
        assert result.attributes["r"] == "50"

    @pytest.mark.asyncio
    async def test_custom_circle(self, context: ProcessingContext):
        node = CircleNode(cx=100, cy=100, radius=75)
        result = await node.process(context)
        assert result.attributes["cx"] == "100"
        assert result.attributes["cy"] == "100"
        assert result.attributes["r"] == "75"


class TestEllipseNode:
    """Tests for EllipseNode SVG shape."""

    @pytest.mark.asyncio
    async def test_default_ellipse(self, context: ProcessingContext):
        node = EllipseNode()
        result = await node.process(context)
        assert result.name == "ellipse"
        assert result.attributes["rx"] == "100"
        assert result.attributes["ry"] == "50"

    @pytest.mark.asyncio
    async def test_custom_ellipse(self, context: ProcessingContext):
        node = EllipseNode(cx=50, cy=50, rx=80, ry=40)
        result = await node.process(context)
        assert result.attributes["cx"] == "50"
        assert result.attributes["cy"] == "50"
        assert result.attributes["rx"] == "80"
        assert result.attributes["ry"] == "40"


class TestLineNode:
    """Tests for LineNode SVG shape."""

    @pytest.mark.asyncio
    async def test_default_line(self, context: ProcessingContext):
        node = LineNode()
        result = await node.process(context)
        assert result.name == "line"
        assert result.attributes["x1"] == "0"
        assert result.attributes["y1"] == "0"
        assert result.attributes["x2"] == "100"
        assert result.attributes["y2"] == "100"

    @pytest.mark.asyncio
    async def test_custom_line(self, context: ProcessingContext):
        node = LineNode(x1=10, y1=20, x2=300, y2=400)
        result = await node.process(context)
        assert result.attributes["x1"] == "10"
        assert result.attributes["y1"] == "20"
        assert result.attributes["x2"] == "300"
        assert result.attributes["y2"] == "400"


class TestPolygonNode:
    """Tests for PolygonNode SVG shape."""

    @pytest.mark.asyncio
    async def test_triangle(self, context: ProcessingContext):
        node = PolygonNode(points="100,10 40,198 190,78")
        result = await node.process(context)
        assert result.name == "polygon"
        assert result.attributes["points"] == "100,10 40,198 190,78"


class TestPathNode:
    """Tests for PathNode SVG shape."""

    @pytest.mark.asyncio
    async def test_path_with_data(self, context: ProcessingContext):
        path_data = "M 10 10 H 90 V 90 H 10 L 10 10"
        node = PathNode(path_data=path_data)
        result = await node.process(context)
        assert result.name == "path"
        assert result.attributes["d"] == path_data


class TestText:
    """Tests for Text SVG element."""

    @pytest.mark.asyncio
    async def test_default_text(self, context: ProcessingContext):
        node = Text(text="Hello World")
        result = await node.process(context)
        assert result.name == "text"
        assert result.content == "Hello World"
        assert result.attributes["font-family"] == "Arial"
        assert result.attributes["font-size"] == "16"

    @pytest.mark.asyncio
    async def test_custom_text(self, context: ProcessingContext):
        node = Text(
            text="Custom Text",
            x=50,
            y=100,
            font_family="Helvetica",
            font_size=24,
            text_anchor=SVGTextAnchor.MIDDLE,
        )
        result = await node.process(context)
        assert result.attributes["x"] == "50"
        assert result.attributes["y"] == "100"
        assert result.attributes["font-family"] == "Helvetica"
        assert result.attributes["font-size"] == "24"
        assert result.attributes["text-anchor"] == "middle"


class TestGaussianBlur:
    """Tests for GaussianBlur filter."""

    @pytest.mark.asyncio
    async def test_default_blur(self, context: ProcessingContext):
        node = GaussianBlur()
        result = await node.process(context)
        assert result.name == "filter"
        assert result.attributes["id"] == "filter_gaussian_blur"
        assert len(result.children) == 1
        assert result.children[0].name == "feGaussianBlur"
        assert result.children[0].attributes["stdDeviation"] == "3.0"

    @pytest.mark.asyncio
    async def test_custom_blur(self, context: ProcessingContext):
        node = GaussianBlur(std_deviation=5.5)
        result = await node.process(context)
        assert result.children[0].attributes["stdDeviation"] == "5.5"


class TestDropShadow:
    """Tests for DropShadow filter."""

    @pytest.mark.asyncio
    async def test_default_shadow(self, context: ProcessingContext):
        node = DropShadow()
        result = await node.process(context)
        assert result.name == "filter"
        assert result.attributes["id"] == "filter_drop_shadow"
        # Should have multiple filter primitives
        assert len(result.children) >= 4

    @pytest.mark.asyncio
    async def test_custom_shadow(self, context: ProcessingContext):
        node = DropShadow(std_deviation=5.0, dx=4, dy=6)
        result = await node.process(context)
        # Verify the blur element
        blur_elem = result.children[0]
        assert blur_elem.attributes["stdDeviation"] == "5.0"
        # Verify the offset element
        offset_elem = result.children[1]
        assert offset_elem.attributes["dx"] == "4"
        assert offset_elem.attributes["dy"] == "6"


class TestDocument:
    """Tests for SVG Document node."""

    @pytest.mark.asyncio
    async def test_empty_document(self, context: ProcessingContext):
        node = Document()
        result = await node.process(context)
        assert result.data is not None
        svg_str = result.data.decode("utf-8")
        assert '<?xml version="1.0"' in svg_str
        assert "xmlns" in svg_str
        assert 'width="800"' in svg_str
        assert 'height="600"' in svg_str

    @pytest.mark.asyncio
    async def test_document_with_content(self, context: ProcessingContext):
        rect = SVGElement(
            name="rect",
            attributes={"x": "10", "y": "10", "width": "100", "height": "50"},
        )
        node = Document(content=[rect], width=400, height=300)
        result = await node.process(context)
        svg_str = result.data.decode("utf-8")
        assert 'width="400"' in svg_str
        assert 'height="300"' in svg_str


class TestGradient:
    """Tests for Gradient SVG element."""

    @pytest.mark.asyncio
    async def test_linear_gradient(self, context: ProcessingContext):
        node = Gradient(
            gradient_type=Gradient.GradientType.LINEAR,
            x1=0,
            y1=0,
            x2=100,
            y2=0,
            color1=ColorRef(value="#ff0000"),
            color2=ColorRef(value="#0000ff"),
        )
        result = await node.process(context)
        assert result.name == "linearGradient"
        assert len(result.children) == 2  # Two stops

    @pytest.mark.asyncio
    async def test_radial_gradient(self, context: ProcessingContext):
        node = Gradient(gradient_type=Gradient.GradientType.RADIAL)
        result = await node.process(context)
        assert result.name == "radialGradient"


class TestTransform:
    """Tests for Transform SVG element."""

    @pytest.mark.asyncio
    async def test_translate(self, context: ProcessingContext):
        rect = SVGElement(name="rect", attributes={"x": "0", "y": "0"})
        node = Transform(content=rect, translate_x=50, translate_y=100)
        result = await node.process(context)
        assert "translate(" in result.attributes.get("transform", "")
        assert "50" in result.attributes.get("transform", "")
        assert "100" in result.attributes.get("transform", "")

    @pytest.mark.asyncio
    async def test_rotate(self, context: ProcessingContext):
        rect = SVGElement(name="rect", attributes={"x": "0", "y": "0"})
        node = Transform(content=rect, rotate=45)
        result = await node.process(context)
        assert "rotate(" in result.attributes.get("transform", "")
        assert "45" in result.attributes.get("transform", "")

    @pytest.mark.asyncio
    async def test_scale(self, context: ProcessingContext):
        rect = SVGElement(name="rect", attributes={"x": "0", "y": "0"})
        node = Transform(content=rect, scale_x=2, scale_y=1.5)
        result = await node.process(context)
        assert "scale(" in result.attributes.get("transform", "")
        assert "1.5" in result.attributes.get("transform", "")

    @pytest.mark.asyncio
    async def test_combined_transforms(self, context: ProcessingContext):
        rect = SVGElement(name="rect", attributes={"x": "0", "y": "0"})
        node = Transform(
            content=rect, translate_x=10, translate_y=20, rotate=30, scale_x=2, scale_y=2
        )
        result = await node.process(context)
        transform = result.attributes.get("transform", "")
        assert "translate(" in transform
        assert "rotate(" in transform
        assert "scale(" in transform

    @pytest.mark.asyncio
    async def test_empty_content(self, context: ProcessingContext):
        node = Transform()
        result = await node.process(context)
        assert result.name == ""  # Empty SVGElement


class TestClipPath:
    """Tests for ClipPath SVG element."""

    @pytest.mark.asyncio
    async def test_clip_path(self, context: ProcessingContext):
        clip_shape = SVGElement(name="circle", attributes={"cx": "50", "cy": "50", "r": "40"})
        content = SVGElement(name="rect", attributes={"x": "0", "y": "0"})
        node = ClipPath(clip_content=clip_shape, content=content)
        result = await node.process(context)
        assert result.name == "g"
        assert len(result.children) == 2

    @pytest.mark.asyncio
    async def test_empty_clip_path(self, context: ProcessingContext):
        node = ClipPath()
        result = await node.process(context)
        # When clip_content or content is empty, it returns a group or empty element
        assert result is not None
