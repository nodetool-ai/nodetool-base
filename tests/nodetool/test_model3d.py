"""
Tests for 3D model nodes.
"""

import pytest

from nodetool.metadata.types import Model3DRef
from nodetool.nodes.nodetool.input import Model3DInput
from nodetool.nodes.nodetool.model3d import (
    FormatConverter,
    GetModel3DMetadata,
    Transform3D,
    Decimate,
    RecalculateNormals,
    CenterMesh,
    FlipNormals,
    MergeMeshes,
    OutputFormat,
    ShadingMode,
)
from nodetool.workflows.processing_context import ProcessingContext


# Create a simple test mesh in OBJ format (cube)
SIMPLE_CUBE_OBJ = b"""# Simple cube
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
v 0.0 0.0 1.0
v 1.0 0.0 1.0
v 1.0 1.0 1.0
v 0.0 1.0 1.0
f 1 2 3 4
f 5 6 7 8
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
"""


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.fixture
def cube_model():
    """Create a Model3DRef with cube data."""
    return Model3DRef(
        data=SIMPLE_CUBE_OBJ,
        format="obj",
    )


@pytest.mark.asyncio
async def test_model3d_input(context):
    """Test Model3DInput node."""
    model = Model3DRef(uri="test.glb", format="glb")
    node = Model3DInput(
        name="model_input",
        value=model,
        description="test",
    )
    result = await node.process(context)
    assert result == model
    assert isinstance(result, Model3DRef)


@pytest.mark.asyncio
async def test_model3d_input_return_type():
    """Test Model3DInput return type."""
    assert Model3DInput.return_type() == Model3DRef


@pytest.mark.asyncio
async def test_format_converter(context, cube_model):
    """Test FormatConverter node."""
    node = FormatConverter(
        model=cube_model,
        output_format=OutputFormat.STL,
    )
    result = await node.process(context)

    assert isinstance(result, Model3DRef)
    assert result.format == "stl"
    assert result.data is not None


@pytest.mark.asyncio
async def test_format_converter_empty_model(context):
    """Test FormatConverter with empty model."""
    node = FormatConverter(
        model=Model3DRef(),
        output_format=OutputFormat.STL,
    )
    with pytest.raises(ValueError, match="input model is not connected"):
        await node.process(context)


@pytest.mark.asyncio
async def test_get_model3d_metadata(context, cube_model):
    """Test GetModel3DMetadata node."""
    node = GetModel3DMetadata(model=cube_model)
    result = await node.process(context)

    assert result["format"] == "obj"
    assert result["vertex_count"] == 8  # cube has 8 vertices
    assert result["face_count"] > 0
    assert "bounds_min" in result
    assert "bounds_max" in result
    assert isinstance(result["bounds_min"], list)
    assert isinstance(result["bounds_max"], list)


@pytest.mark.asyncio
async def test_get_model3d_metadata_empty_model(context):
    """Test GetModel3DMetadata with empty model."""
    node = GetModel3DMetadata(model=Model3DRef())
    with pytest.raises(ValueError, match="input model is not connected"):
        await node.process(context)


@pytest.mark.asyncio
async def test_transform3d_translate(context, cube_model):
    """Test Transform3D translation."""
    node = Transform3D(
        model=cube_model,
        translate_x=10.0,
        translate_y=5.0,
        translate_z=2.0,
    )
    result = await node.process(context)

    assert isinstance(result, Model3DRef)
    assert result.data is not None


@pytest.mark.asyncio
async def test_transform3d_rotate(context, cube_model):
    """Test Transform3D rotation."""
    node = Transform3D(
        model=cube_model,
        rotate_x=45.0,
        rotate_y=90.0,
        rotate_z=0.0,
    )
    result = await node.process(context)

    assert isinstance(result, Model3DRef)
    assert result.data is not None


@pytest.mark.asyncio
async def test_transform3d_scale(context, cube_model):
    """Test Transform3D scaling."""
    node = Transform3D(
        model=cube_model,
        scale_x=2.0,
        scale_y=2.0,
        scale_z=2.0,
        uniform_scale=1.0,
    )
    result = await node.process(context)

    assert isinstance(result, Model3DRef)
    assert result.data is not None


@pytest.mark.asyncio
async def test_transform3d_empty_model(context):
    """Test Transform3D with empty model."""
    node = Transform3D(model=Model3DRef())
    with pytest.raises(ValueError, match="input model is not connected"):
        await node.process(context)


@pytest.mark.asyncio
async def test_decimate(context, cube_model):
    """Test Decimate node."""
    node = Decimate(
        model=cube_model,
        target_ratio=0.5,
    )
    result = await node.process(context)

    assert isinstance(result, Model3DRef)
    assert result.data is not None
    assert result.metadata is not None
    assert "original_faces" in result.metadata


@pytest.mark.asyncio
async def test_decimate_empty_model(context):
    """Test Decimate with empty model."""
    node = Decimate(model=Model3DRef())
    with pytest.raises(ValueError, match="input model is not connected"):
        await node.process(context)


@pytest.mark.asyncio
async def test_recalculate_normals_smooth(context, cube_model):
    """Test RecalculateNormals with smooth mode."""
    node = RecalculateNormals(
        model=cube_model,
        mode=ShadingMode.SMOOTH,
    )
    result = await node.process(context)

    assert isinstance(result, Model3DRef)
    assert result.data is not None


@pytest.mark.asyncio
async def test_recalculate_normals_flat(context, cube_model):
    """Test RecalculateNormals with flat mode."""
    node = RecalculateNormals(
        model=cube_model,
        mode=ShadingMode.FLAT,
    )
    result = await node.process(context)

    assert isinstance(result, Model3DRef)
    assert result.data is not None


@pytest.mark.asyncio
async def test_recalculate_normals_auto(context, cube_model):
    """Test RecalculateNormals with auto mode."""
    node = RecalculateNormals(
        model=cube_model,
        mode=ShadingMode.AUTO,
    )
    result = await node.process(context)

    assert isinstance(result, Model3DRef)
    assert result.data is not None


@pytest.mark.asyncio
async def test_recalculate_normals_empty_model(context):
    """Test RecalculateNormals with empty model."""
    node = RecalculateNormals(model=Model3DRef())
    with pytest.raises(ValueError, match="input model is not connected"):
        await node.process(context)


@pytest.mark.asyncio
async def test_center_mesh_centroid(context, cube_model):
    """Test CenterMesh with centroid option."""
    node = CenterMesh(
        model=cube_model,
        use_centroid=True,
    )
    result = await node.process(context)

    assert isinstance(result, Model3DRef)
    assert result.data is not None


@pytest.mark.asyncio
async def test_center_mesh_bounding_box(context, cube_model):
    """Test CenterMesh with bounding box center."""
    node = CenterMesh(
        model=cube_model,
        use_centroid=False,
    )
    result = await node.process(context)

    assert isinstance(result, Model3DRef)
    assert result.data is not None


@pytest.mark.asyncio
async def test_center_mesh_empty_model(context):
    """Test CenterMesh with empty model."""
    node = CenterMesh(model=Model3DRef())
    with pytest.raises(ValueError, match="input model is not connected"):
        await node.process(context)


@pytest.mark.asyncio
async def test_flip_normals(context, cube_model):
    """Test FlipNormals node."""
    node = FlipNormals(model=cube_model)
    result = await node.process(context)

    assert isinstance(result, Model3DRef)
    assert result.data is not None


@pytest.mark.asyncio
async def test_flip_normals_empty_model(context):
    """Test FlipNormals with empty model."""
    node = FlipNormals(model=Model3DRef())
    with pytest.raises(ValueError, match="input model is not connected"):
        await node.process(context)


@pytest.mark.asyncio
async def test_merge_meshes(context, cube_model):
    """Test MergeMeshes node."""
    # Create two cubes
    cube1 = Model3DRef(data=SIMPLE_CUBE_OBJ, format="obj")
    cube2 = Model3DRef(data=SIMPLE_CUBE_OBJ, format="obj")

    node = MergeMeshes(models=[cube1, cube2])
    result = await node.process(context)

    assert isinstance(result, Model3DRef)
    assert result.data is not None
    assert result.metadata is not None
    assert result.metadata["mesh_count"] == 2


@pytest.mark.asyncio
async def test_merge_meshes_empty_list(context):
    """Test MergeMeshes with empty list."""
    node = MergeMeshes(models=[])
    with pytest.raises(ValueError, match="No models provided"):
        await node.process(context)


@pytest.mark.asyncio
async def test_merge_meshes_all_empty_models(context):
    """Test MergeMeshes with all empty models."""
    node = MergeMeshes(models=[Model3DRef(), Model3DRef()])
    with pytest.raises(ValueError, match="No valid meshes"):
        await node.process(context)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output_format,expected_format",
    [
        (OutputFormat.GLB, "glb"),
        (OutputFormat.OBJ, "obj"),
        (OutputFormat.STL, "stl"),
        (OutputFormat.PLY, "ply"),
    ],
)
async def test_format_converter_formats(context, cube_model, output_format, expected_format):
    """Test FormatConverter with different output formats."""
    node = FormatConverter(
        model=cube_model,
        output_format=output_format,
    )
    result = await node.process(context)

    assert isinstance(result, Model3DRef)
    assert result.format == expected_format


# JSON schema tests
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "node_class",
    [
        FormatConverter,
        GetModel3DMetadata,
        Transform3D,
        Decimate,
        RecalculateNormals,
        CenterMesh,
        FlipNormals,
        MergeMeshes,
    ],
)
async def test_node_json_schema(node_class):
    """Test that all nodes have valid JSON schema."""
    node = node_class()
    schema = node.get_json_schema()
    assert isinstance(schema, dict)
    assert "type" in schema
    assert "properties" in schema


# Tests for TextTo3D and ImageTo3D nodes
from unittest.mock import AsyncMock, MagicMock, patch

from nodetool.metadata.types import ImageRef, Model3DModel, Provider
from nodetool.nodes.nodetool.model3d import TextTo3D, ImageTo3D, OutputFormat


# Sample GLB bytes (minimal valid GLB header)
SAMPLE_GLB_BYTES = b'glTF\x02\x00\x00\x00\x00\x00\x00\x00'


@pytest.fixture
def mock_provider():
    """Create a mock provider for 3D generation."""
    provider = AsyncMock()
    provider.text_to_3d = AsyncMock(return_value=SAMPLE_GLB_BYTES)
    provider.image_to_3d = AsyncMock(return_value=SAMPLE_GLB_BYTES)
    return provider


@pytest.fixture
def sample_image():
    """Create a sample ImageRef for testing."""
    import io
    import PIL.Image
    buffer = io.BytesIO()
    PIL.Image.new("RGB", (100, 100), color="blue").save(buffer, format="PNG")
    return ImageRef(data=buffer.getvalue())


@pytest.mark.asyncio
async def test_text_to_3d_empty_prompt(context):
    """Test TextTo3D with empty prompt raises error."""
    node = TextTo3D(
        model=Model3DModel(provider=Provider.Meshy, id="meshy-4", name="Test"),
        prompt="",
    )
    with pytest.raises(ValueError, match="Prompt is required"):
        await node.process(context)


@pytest.mark.asyncio
async def test_text_to_3d_basic_fields():
    """Test TextTo3D basic fields."""
    fields = TextTo3D.get_basic_fields()
    assert "model" in fields
    assert "prompt" in fields
    assert "output_format" in fields
    assert "seed" in fields


@pytest.mark.asyncio
async def test_image_to_3d_empty_image(context):
    """Test ImageTo3D with empty image raises error."""
    node = ImageTo3D(
        model=Model3DModel(provider=Provider.Meshy, id="meshy-4-image", name="Test"),
        image=ImageRef(),
    )
    with pytest.raises(ValueError, match="Input image must be connected"):
        await node.process(context)


@pytest.mark.asyncio
async def test_image_to_3d_basic_fields():
    """Test ImageTo3D basic fields."""
    fields = ImageTo3D.get_basic_fields()
    assert "model" in fields
    assert "image" in fields
    assert "output_format" in fields
    assert "seed" in fields


@pytest.mark.asyncio
async def test_text_to_3d_with_mock_provider(context, mock_provider, monkeypatch):
    """Test TextTo3D node with mocked provider."""
    # Mock context.get_provider to return our mock
    async def mock_get_provider(provider):
        return mock_provider
    
    # Mock model3d_from_bytes to return a Model3DRef
    async def mock_model3d_from_bytes(data, name=None, format=None):
        return Model3DRef(data=data, format=format)
    
    monkeypatch.setattr(context, "get_provider", mock_get_provider)
    monkeypatch.setattr(context, "model3d_from_bytes", mock_model3d_from_bytes)
    
    node = TextTo3D(
        model=Model3DModel(provider=Provider.Meshy, id="meshy-4", name="Test"),
        prompt="A red cube",
        output_format=OutputFormat.GLB,
    )
    
    result = await node.process(context)
    
    assert isinstance(result, Model3DRef)
    assert result.format == "glb"
    mock_provider.text_to_3d.assert_called_once()


@pytest.mark.asyncio
async def test_image_to_3d_with_mock_provider(context, mock_provider, sample_image, monkeypatch):
    """Test ImageTo3D node with mocked provider."""
    # Mock context.get_provider to return our mock
    async def mock_get_provider(provider):
        return mock_provider
    
    # Mock asset_to_io to return image bytes
    async def mock_asset_to_io(asset):
        import io
        return io.BytesIO(sample_image.data)
    
    # Mock model3d_from_bytes to return a Model3DRef
    async def mock_model3d_from_bytes(data, name=None, format=None):
        return Model3DRef(data=data, format=format)
    
    monkeypatch.setattr(context, "get_provider", mock_get_provider)
    monkeypatch.setattr(context, "asset_to_io", mock_asset_to_io)
    monkeypatch.setattr(context, "model3d_from_bytes", mock_model3d_from_bytes)
    
    node = ImageTo3D(
        model=Model3DModel(provider=Provider.Meshy, id="meshy-4-image", name="Test"),
        image=sample_image,
        output_format=OutputFormat.GLB,
    )
    
    result = await node.process(context)
    
    assert isinstance(result, Model3DRef)
    assert result.format == "glb"
    mock_provider.image_to_3d.assert_called_once()


# JSON schema tests for new nodes
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "node_class",
    [
        TextTo3D,
        ImageTo3D,
    ],
)
async def test_generation_node_json_schema(node_class):
    """Test that generation nodes have valid JSON schema."""
    node = node_class()
    schema = node.get_json_schema()
    assert isinstance(schema, dict)
    assert "type" in schema
    assert "properties" in schema
