from nodetool.nodes.lib.numpy.arithmetic import (
    AddArray,
    DivideArray,
    ModulusArray,
    MultiplyArray,
    SubtractArray,
)
from nodetool.nodes.lib.numpy.manipulation import (
    IndexArray,
    MatMul,
    SliceArray,
    SplitArray,
    Stack,
    TransposeArray,
)
from nodetool.nodes.lib.numpy.math import (
    AbsArray,
    CosineArray,
    ExpArray,
    LogArray,
    PowerArray,
    SineArray,
    SqrtArray,
)
from nodetool.nodes.lib.numpy.conversion import (
    ConvertToArray,
    ConvertToImage,
    ConvertToAudio,
    ArrayToScalar,
    ScalarToArray,
    ListToArray,
    ArrayToList,
)
from nodetool.nodes.lib.numpy.reshaping import (
    Reshape1D,
    Reshape2D,
    Reshape3D,
    Reshape4D,
)
from nodetool.nodes.lib.numpy.statistics import (
    MaxArray,
    MinArray,
    MeanArray,
    SumArray,
    ArgMaxArray,
    ArgMinArray,
)
from nodetool.nodes.lib.numpy.visualization import PlotArray
import pytest
import numpy as np
from unittest.mock import patch
from PIL import Image
import io

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import NPArray, ImageRef, AudioRef


@pytest.fixture
def processing_context():
    # Using test values for required parameters
    return ProcessingContext(user_id="test-user", auth_token="test-token")


@pytest.fixture
def sample_array():
    return NPArray.from_numpy(np.array([1, 2, 3, 4]))


@pytest.fixture
def sample_2d_array():
    return NPArray.from_numpy(np.array([[1, 2], [3, 4]]))


class TestBinaryOperations:
    @pytest.mark.parametrize(
        "NodeClass, a, b, expected",
        [
            (AddArray, 2, 3, 5),
            (SubtractArray, 5, 3, 2),
            (MultiplyArray, 2, 3, 6),
            (DivideArray, 6, 3, 2),
            (ModulusArray, 7, 3, 1),
        ],
    )
    @pytest.mark.asyncio
    async def test_basic_math_operations(
        self, processing_context, NodeClass, a, b, expected
    ):
        node = NodeClass(a=a, b=b)
        result = await node.process(processing_context)
        assert result == expected

    @pytest.mark.parametrize(
        "node, expected_type",
        [
            (AddArray(a=5, b=5), (float, int, NPArray)),
            (
                AddArray(
                    a=NPArray.from_numpy(np.array([1, 2])),
                    b=NPArray.from_numpy(np.array([3, 4])),
                ),
                NPArray,
            ),
            (SubtractArray(a=5, b=5), (float, int, NPArray)),
            (MultiplyArray(a=5, b=5), (float, int, NPArray)),
            (DivideArray(a=5, b=5), (float, int, NPArray)),
            (ModulusArray(a=5, b=5), (float, int, NPArray)),
        ],
    )
    @pytest.mark.asyncio
    async def test_math_nodes_types(self, processing_context, node, expected_type):
        try:
            result = await node.process(processing_context)
            assert isinstance(result, expected_type)
        except Exception as e:
            pytest.fail(f"Error processing {node.__class__.__name__}: {str(e)}")


class TestTrigonometricOperations:
    @pytest.mark.parametrize(
        "NodeClass, input_value, expected",
        [
            (SineArray, 0, 0),
            (CosineArray, np.pi / 2, 0),
            (CosineArray, 0, 1),
            (CosineArray, np.pi, -1),
        ],
    )
    @pytest.mark.asyncio
    async def test_trig_functions(
        self, processing_context, NodeClass, input_value, expected
    ):
        node = NodeClass(angle_rad=input_value)
        result = await node.process(processing_context)
        assert np.isclose(result, expected)


class TestMathOperations:
    @pytest.mark.asyncio
    async def test_power_array(self, processing_context, sample_array):
        # Setup
        node = PowerArray(base=sample_array, exponent=2)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([1, 4, 9, 16]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_sqrt_array(self, processing_context):
        # Setup
        node = SqrtArray(values=NPArray.from_numpy(np.array([4, 9, 16, 25])))

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([2, 3, 4, 5]), result.to_numpy())


class TestConversionOperations:
    @pytest.mark.asyncio
    async def test_convert_to_array(self, processing_context):
        # Setup
        # Create a real PIL image
        test_image = Image.new("RGB", (10, 10), color=(255, 255, 255))
        img_byte_arr = io.BytesIO()
        test_image.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()

        # Create ImageRef with actual image data
        image_ref = ImageRef(data=img_bytes)
        node = ConvertToArray(image=image_ref)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        assert result.to_numpy().shape == (10, 10, 3)
        assert np.all(result.to_numpy() == 1.0)  # Normalized to [0,1]

    @pytest.mark.asyncio
    async def test_convert_to_image(self, processing_context):
        # Setup
        array_data = NPArray.from_numpy(np.ones((10, 10, 3)))
        node = ConvertToImage(values=array_data)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, ImageRef)
        # ImageRef may have data or a URI - either is valid
        assert result.data is not None or result.uri is not None

    @pytest.mark.asyncio
    async def test_convert_to_audio(self, processing_context):
        # Setup
        array_data = NPArray.from_numpy(np.ones(1000))
        node = ConvertToAudio(values=array_data, sample_rate=44100)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, AudioRef)
        assert result.data is not None  # Should have binary data

    @pytest.mark.asyncio
    async def test_convert_to_image_with_mock(self, processing_context, mocker):
        # Setup mock
        mocker.patch.object(
            processing_context,
            "image_from_pil",
            return_value=ImageRef(asset_id="test_image_id"),
        )

        array_data = NPArray.from_numpy(np.random.rand(100, 100, 3))
        node = ConvertToImage(values=array_data)
        result = await node.process(processing_context)

        assert isinstance(result, ImageRef)
        assert result.asset_id == "test_image_id"


class TestArrayManipulation:
    @pytest.mark.asyncio
    async def test_stack(self, processing_context):
        # Setup
        arrays = [
            NPArray.from_numpy(np.array([1, 2, 3])),
            NPArray.from_numpy(np.array([4, 5, 6])),
        ]
        node = Stack(arrays=arrays, axis=0)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        expected = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(expected, result.to_numpy())

    @pytest.mark.asyncio
    async def test_matmul(self, processing_context):
        # Setup
        a = NPArray.from_numpy(np.array([[1, 2], [3, 4]]))
        b = NPArray.from_numpy(np.array([[5, 6], [7, 8]]))
        node = MatMul(a=a, b=b)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(expected, result.to_numpy())

    @pytest.mark.asyncio
    async def test_transpose_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
        node = TransposeArray(values=array)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(expected, result.to_numpy())


class TestReductionOperations:
    @pytest.mark.asyncio
    async def test_max_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
        node = MaxArray(values=array, axis=1)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([3, 6]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_min_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
        node = MinArray(values=array, axis=1)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([1, 4]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_mean_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
        node = MeanArray(values=array, axis=1)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([2, 5]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_sum_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
        node = SumArray(values=array, axis=1)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([6, 15]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_argmax_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([[1, 2, 3], [6, 5, 4]]))
        node = ArgMaxArray(values=array, axis=1)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([2, 0]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_argmin_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([[1, 2, 3], [6, 5, 4]]))
        node = ArgMinArray(values=array, axis=1)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([0, 2]), result.to_numpy())


class TestElementWiseOperations:
    @pytest.mark.asyncio
    async def test_abs_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([-1, -2, 3, -4]))
        node = AbsArray(values=array)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([1, 2, 3, 4]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_exp_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([0, 1, 2]))
        node = ExpArray(values=array)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        expected = np.array([1, np.e, np.e**2])
        np.testing.assert_allclose(expected, result.to_numpy())

    @pytest.mark.asyncio
    async def test_log_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([1, np.e, np.e**2]))
        node = LogArray(values=array)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        expected = np.array([0, 1, 2])
        np.testing.assert_allclose(expected, result.to_numpy())


class TestTypeConversions:
    @pytest.mark.asyncio
    async def test_array_to_scalar(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([42]))
        node = ArrayToScalar(values=array)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, int)
        assert result == 42

    @pytest.mark.asyncio
    async def test_scalar_to_array(self, processing_context):
        # Setup
        node = ScalarToArray(value=42)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([42]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_list_to_array(self, processing_context):
        # Setup
        node = ListToArray(values=[1, 2, 3, 4])

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([1, 2, 3, 4]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_array_to_list(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([1, 2, 3, 4]))
        node = ArrayToList(values=array)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, list)
        assert result == [1, 2, 3, 4]


class TestVisualization:
    @pytest.mark.asyncio
    async def test_plot_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([1, 2, 3, 4]))
        node = PlotArray(values=array, plot_type=PlotArray.PlotType.LINE)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, ImageRef)
        assert result.data is not None  # Should have binary data
        # Optional: could verify it's a valid image by loading with PIL
        img = Image.open(io.BytesIO(result.data))
        assert img.format in ("PNG", "JPEG")


class TestSlicingAndIndexing:
    @pytest.mark.asyncio
    async def test_slice_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        node = SliceArray(values=array, start=2, stop=8, step=2, axis=0)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([2, 4, 6]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_index_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([10, 20, 30, 40, 50]))

        # Need to patch the map function since it's used in the implementation
        with patch("nodetool.nodes.lib.numpy.map", return_value=[1, 3]):
            node = IndexArray(values=array, indices="1,3", axis=0)

            # Execute
            result = await node.process(processing_context)

            # Assert
            assert isinstance(result, NPArray)
            np.testing.assert_array_equal(np.array([20, 40]), result.to_numpy())


class TestReshaping:
    @pytest.mark.asyncio
    async def test_reshape_1d(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([[1, 2], [3, 4]]))
        node = Reshape1D(values=array, num_elements=4)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([1, 2, 3, 4]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_reshape_2d(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([1, 2, 3, 4, 5, 6]))
        node = Reshape2D(values=array, num_rows=2, num_cols=3)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        expected = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(expected, result.to_numpy())

    @pytest.mark.asyncio
    async def test_reshape_3d(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array(range(24)))
        node = Reshape3D(values=array, num_rows=2, num_cols=3, num_depths=4)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        assert result.to_numpy().shape == (2, 3, 4)

    @pytest.mark.asyncio
    async def test_reshape_4d(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array(range(24)))
        node = Reshape4D(
            values=array, num_rows=2, num_cols=3, num_depths=2, num_channels=2
        )

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        assert result.to_numpy().shape == (2, 3, 2, 2)


class TestSplitting:
    @pytest.mark.asyncio
    async def test_split_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([1, 2, 3, 4, 5, 6]))
        node = SplitArray(values=array, num_splits=3, axis=0)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(r, NPArray) for r in result)
        np.testing.assert_array_equal(np.array([1, 2]), result[0].to_numpy())
        np.testing.assert_array_equal(np.array([3, 4]), result[1].to_numpy())
        np.testing.assert_array_equal(np.array([5, 6]), result[2].to_numpy())


@pytest.mark.parametrize(
    "NodeClass",
    [
        AddArray,
        SubtractArray,
        MultiplyArray,
        DivideArray,
        ModulusArray,
        SineArray,
        CosineArray,
        PowerArray,
        SqrtArray,
    ],
)
def test_node_attributes(NodeClass):
    node = NodeClass()
    assert hasattr(node, "process")
    assert callable(node.process)
