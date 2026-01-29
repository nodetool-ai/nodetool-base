import pytest
from unittest.mock import AsyncMock, MagicMock
from PIL import Image
import numpy as np
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import ImageRef
from nodetool.nodes.lib.grid import (
    Tile,
    make_grid,
    create_gradient_mask,
    combine_grid,
    SliceImageGrid,
    CombineImageGrid,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.fixture
def mock_context():
    """Create a mock context for testing image operations."""
    ctx = MagicMock(spec=ProcessingContext)
    return ctx


class TestTile:
    """Tests for Tile class."""

    def test_tile_creation(self):
        image = Image.new("RGB", (100, 100))
        tile = Tile(image, x=50, y=75)
        
        assert tile.image is image
        assert tile.x == 50
        assert tile.y == 75


class TestMakeGrid:
    """Tests for make_grid function."""

    def test_make_grid_basic(self):
        tiles, cols, rows = make_grid(width=1024, height=768, tile_w=512, tile_h=512, overlap=0)
        
        assert cols == 2  # 1024 / 512 = 2
        assert rows == 1  # 768 / 512 = 1 (only fits once)
        assert len(tiles) == 1  # 1 row
        assert len(tiles[0]) == 2  # 2 columns
        
        # Check coordinates
        assert tiles[0][0] == (0, 0)
        assert tiles[0][1] == (512, 0)

    def test_make_grid_with_overlap(self):
        tiles, cols, rows = make_grid(width=1000, height=1000, tile_w=512, tile_h=512, overlap=64)
        
        # With overlap, tiles start at 0, 512-64=448, 448+448=896...
        # First tile at x=0, second at x=448, third would be at x=896 (896+512=1408 > 1000)
        # So we have 2 columns
        assert len(tiles) > 0
        assert len(tiles[0]) >= 1

    def test_make_grid_exact_fit(self):
        tiles, cols, rows = make_grid(width=512, height=512, tile_w=256, tile_h=256, overlap=0)
        
        assert cols == 2
        assert rows == 2
        assert len(tiles) == 2
        assert len(tiles[0]) == 2

    def test_make_grid_no_fit_raises(self):
        with pytest.raises(AssertionError, match="Dimensions must be positive"):
            make_grid(width=0, height=100, tile_w=50, tile_h=50, overlap=0)

    def test_make_grid_negative_tile_size_raises(self):
        with pytest.raises(AssertionError, match="Tile size must be positive"):
            make_grid(width=100, height=100, tile_w=0, tile_h=50, overlap=0)

    def test_make_grid_negative_overlap_raises(self):
        with pytest.raises(AssertionError, match="Overlap must be non-negative"):
            make_grid(width=100, height=100, tile_w=50, tile_h=50, overlap=-1)

    def test_make_grid_single_tile(self):
        tiles, cols, rows = make_grid(width=256, height=256, tile_w=256, tile_h=256, overlap=0)
        
        assert cols == 1
        assert rows == 1
        assert tiles[0][0] == (0, 0)


class TestCreateGradientMask:
    """Tests for create_gradient_mask function."""

    def test_create_gradient_mask_basic(self):
        mask = create_gradient_mask(tile_w=256, tile_h=256, overlap=32)
        
        assert mask.mode == "L"  # Grayscale
        assert mask.size == (256, 256)

    def test_create_gradient_mask_overlap_region(self):
        tile_w, tile_h, overlap = 100, 100, 20
        mask = create_gradient_mask(tile_w, tile_h, overlap)
        
        # The gradient region should be in the last 'overlap' columns/rows
        pixels = np.array(mask)
        
        # Check that the mask has expected size
        assert pixels.shape == (tile_h, tile_w)


class TestCombineGrid:
    """Tests for combine_grid function."""

    def test_combine_grid_basic(self):
        # Create a 2x2 grid of tiles
        tile_w, tile_h = 100, 100
        tiles = []
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
        ]
        
        for row_idx in range(2):
            row = []
            for col_idx in range(2):
                color = colors[row_idx * 2 + col_idx]
                img = Image.new("RGB", (tile_w, tile_h), color)
                tile = Tile(img, x=col_idx * tile_w, y=row_idx * tile_h)
                row.append(tile)
            tiles.append(row)
        
        result = combine_grid(tiles, tile_w, tile_h, width=200, height=200, overlap=0)
        
        assert result.size == (200, 200)
        assert result.mode == "RGB"


class TestSliceImageGrid:
    """Tests for SliceImageGrid node."""

    @pytest.mark.asyncio
    async def test_slice_image_default_3x3(self, mock_context):
        # Create test image
        test_image = Image.new("RGB", (300, 300), (255, 0, 0))
        mock_context.image_to_pil = AsyncMock(return_value=test_image)
        mock_context.image_from_pil = AsyncMock(side_effect=lambda img: ImageRef(uri=f"slice_{id(img)}"))
        
        node = SliceImageGrid(image=ImageRef(uri="test.png"), columns=0, rows=0)
        result = await node.process(mock_context)
        
        # Default 3x3 = 9 tiles
        assert len(result) == 9

    @pytest.mark.asyncio
    async def test_slice_image_with_columns(self, mock_context):
        test_image = Image.new("RGB", (200, 200), (0, 255, 0))
        mock_context.image_to_pil = AsyncMock(return_value=test_image)
        mock_context.image_from_pil = AsyncMock(side_effect=lambda img: ImageRef(uri=f"slice_{id(img)}"))
        
        node = SliceImageGrid(image=ImageRef(uri="test.png"), columns=2, rows=0)
        result = await node.process(mock_context)
        
        # 2 columns, rows calculated to maintain aspect ratio (square image -> 2 rows)
        assert len(result) >= 4

    @pytest.mark.asyncio
    async def test_slice_image_with_rows(self, mock_context):
        test_image = Image.new("RGB", (200, 200), (0, 0, 255))
        mock_context.image_to_pil = AsyncMock(return_value=test_image)
        mock_context.image_from_pil = AsyncMock(side_effect=lambda img: ImageRef(uri=f"slice_{id(img)}"))
        
        node = SliceImageGrid(image=ImageRef(uri="test.png"), columns=0, rows=2)
        result = await node.process(mock_context)
        
        # 2 rows, columns calculated
        assert len(result) >= 4

    @pytest.mark.asyncio
    async def test_slice_image_explicit_grid(self, mock_context):
        test_image = Image.new("RGB", (400, 200), (255, 255, 0))
        mock_context.image_to_pil = AsyncMock(return_value=test_image)
        mock_context.image_from_pil = AsyncMock(side_effect=lambda img: ImageRef(uri=f"slice_{id(img)}"))
        
        node = SliceImageGrid(image=ImageRef(uri="test.png"), columns=4, rows=2)
        result = await node.process(mock_context)
        
        # 4 columns x 2 rows = 8 tiles
        assert len(result) == 8


class TestCombineImageGrid:
    """Tests for CombineImageGrid node."""

    @pytest.mark.asyncio
    async def test_combine_empty_tiles_raises(self, mock_context):
        node = CombineImageGrid(tiles=[], columns=2)
        
        with pytest.raises(ValueError, match="No tiles provided"):
            await node.process(mock_context)

    @pytest.mark.asyncio
    async def test_combine_tiles_default_columns(self, mock_context):
        # Create 4 test tiles
        tile_images = [
            Image.new("RGBA", (50, 50), (255, 0, 0, 255)),
            Image.new("RGBA", (50, 50), (0, 255, 0, 255)),
            Image.new("RGBA", (50, 50), (0, 0, 255, 255)),
            Image.new("RGBA", (50, 50), (255, 255, 0, 255)),
        ]
        
        mock_context.image_to_pil = AsyncMock(side_effect=lambda ref: tile_images[int(ref.uri.split("_")[1])])
        mock_context.image_from_pil = AsyncMock(return_value=ImageRef(uri="combined.png"))
        
        tiles = [ImageRef(uri=f"tile_{i}") for i in range(4)]
        node = CombineImageGrid(tiles=tiles, columns=0)  # Default columns
        
        result = await node.process(mock_context)
        
        assert isinstance(result, ImageRef)
        mock_context.image_from_pil.assert_called_once()

    @pytest.mark.asyncio
    async def test_combine_tiles_with_columns(self, mock_context):
        # Create 6 test tiles
        tile_images = [Image.new("RGBA", (50, 50), (i * 40, i * 40, i * 40, 255)) for i in range(6)]
        
        async def mock_image_to_pil(ref):
            idx = int(ref.uri.split("_")[1])
            return tile_images[idx]
        
        mock_context.image_to_pil = AsyncMock(side_effect=mock_image_to_pil)
        mock_context.image_from_pil = AsyncMock(return_value=ImageRef(uri="combined.png"))
        
        tiles = [ImageRef(uri=f"tile_{i}") for i in range(6)]
        node = CombineImageGrid(tiles=tiles, columns=3)  # 3 columns = 2 rows
        
        await node.process(mock_context)
        
        # Verify that image_from_pil was called with the combined image
        assert mock_context.image_from_pil.call_count == 1
        
        # Get the combined image passed to image_from_pil
        combined_image = mock_context.image_from_pil.call_args[0][0]
        
        # Expected size: 3 columns x 50 = 150 width, 2 rows x 50 = 100 height
        assert combined_image.size == (150, 100)

    @pytest.mark.asyncio
    async def test_combine_tiles_different_sizes(self, mock_context):
        """Test combining tiles of different sizes (uses max width/height)."""
        tile_images = [
            Image.new("RGBA", (40, 40), (255, 0, 0, 255)),
            Image.new("RGBA", (60, 40), (0, 255, 0, 255)),  # Wider
            Image.new("RGBA", (40, 60), (0, 0, 255, 255)),  # Taller
            Image.new("RGBA", (50, 50), (255, 255, 0, 255)),
        ]
        
        async def mock_image_to_pil(ref):
            idx = int(ref.uri.split("_")[1])
            return tile_images[idx]
        
        mock_context.image_to_pil = AsyncMock(side_effect=mock_image_to_pil)
        mock_context.image_from_pil = AsyncMock(return_value=ImageRef(uri="combined.png"))
        
        tiles = [ImageRef(uri=f"tile_{i}") for i in range(4)]
        node = CombineImageGrid(tiles=tiles, columns=2)
        
        await node.process(mock_context)
        
        # Get the combined image
        combined_image = mock_context.image_from_pil.call_args[0][0]
        
        # Should use max dimensions: 60 width, 60 height per tile
        # 2 columns x 60 = 120, 2 rows x 60 = 120
        assert combined_image.size == (120, 120)


class TestGridIntegration:
    """Integration tests for slice and combine operations."""

    @pytest.mark.asyncio
    async def test_slice_and_recombine(self, mock_context):
        """Test slicing an image and recombining it back."""
        original_image = Image.new("RGB", (100, 100), (128, 128, 128))
        
        # Setup for slicing
        sliced_images = []
        mock_context.image_to_pil = AsyncMock(return_value=original_image)
        
        async def mock_image_from_pil_slice(img):
            sliced_images.append(img)
            return ImageRef(uri=f"slice_{len(sliced_images) - 1}")
        
        mock_context.image_from_pil = AsyncMock(side_effect=mock_image_from_pil_slice)
        
        # Slice into 2x2 grid
        slice_node = SliceImageGrid(image=ImageRef(uri="original.png"), columns=2, rows=2)
        tiles = await slice_node.process(mock_context)
        
        assert len(tiles) == 4
        assert len(sliced_images) == 4
        
        # Now recombine
        async def mock_image_to_pil_combine(ref):
            idx = int(ref.uri.split("_")[1])
            return sliced_images[idx].convert("RGBA")
        
        mock_context.image_to_pil = AsyncMock(side_effect=mock_image_to_pil_combine)
        mock_context.image_from_pil = AsyncMock(return_value=ImageRef(uri="recombined.png"))
        
        combine_node = CombineImageGrid(tiles=tiles, columns=2)
        await combine_node.process(mock_context)
        
        # Verify combined image dimensions
        combined_image = mock_context.image_from_pil.call_args[0][0]
        assert combined_image.size == (100, 100)
