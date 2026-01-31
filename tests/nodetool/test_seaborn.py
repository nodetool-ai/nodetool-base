import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import (
    ImageRef,
    ChartConfig,
    ChartData,
    DataSeries,
    DataframeRef,
    ColumnDef,
    SeabornPlotType,
)
from nodetool.nodes.lib.seaborn import (
    ChartRenderer,
    SeabornStyle,
    SeabornContext,
    SeabornPalette,
    SeabornFont,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return DataframeRef(
        data=[
            [1, 10, "A"],
            [2, 20, "A"],
            [3, 15, "B"],
            [4, 25, "B"],
            [5, 30, "A"],
        ],
        columns=[
            ColumnDef(name="x", data_type="int"),
            ColumnDef(name="y", data_type="int"),
            ColumnDef(name="category", data_type="string"),
        ],
    )


@pytest.mark.asyncio
async def test_chart_renderer_scatterplot(context: ProcessingContext, sample_dataframe):
    """Test ChartRenderer with scatterplot."""
    chart_config = ChartConfig(
        title="Test Scatter Plot",
        data=ChartData(
            series=[
                DataSeries(
                    x="x",
                    y="y",
                    plot_type=SeabornPlotType.SCATTER,
                )
            ]
        ),
    )
    node = ChartRenderer(
        chart_config=chart_config,
        data=sample_dataframe,
        width=400,
        height=300,
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)
    img = await context.image_to_pil(result)
    assert img.width > 0
    assert img.height > 0


@pytest.mark.asyncio
async def test_chart_renderer_lineplot(context: ProcessingContext, sample_dataframe):
    """Test ChartRenderer with lineplot."""
    chart_config = ChartConfig(
        title="Test Line Plot",
        data=ChartData(
            series=[
                DataSeries(
                    x="x",
                    y="y",
                    plot_type=SeabornPlotType.LINE,
                )
            ]
        ),
    )
    node = ChartRenderer(
        chart_config=chart_config,
        data=sample_dataframe,
        width=400,
        height=300,
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_chart_renderer_barplot(context: ProcessingContext, sample_dataframe):
    """Test ChartRenderer with barplot."""
    chart_config = ChartConfig(
        title="Test Bar Plot",
        data=ChartData(
            series=[
                DataSeries(
                    x="category",
                    y="y",
                    plot_type=SeabornPlotType.BARPLOT,
                )
            ]
        ),
    )
    node = ChartRenderer(
        chart_config=chart_config,
        data=sample_dataframe,
        width=400,
        height=300,
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
@pytest.mark.skip(reason="Seaborn boxplot alpha parameter issue in library code")
async def test_chart_renderer_boxplot(context: ProcessingContext, sample_dataframe):
    """Test ChartRenderer with boxplot."""
    chart_config = ChartConfig(
        title="Test Box Plot",
        data=ChartData(
            series=[
                DataSeries(
                    x="category",
                    y="y",
                    plot_type=SeabornPlotType.BOXPLOT,
                )
            ]
        ),
    )
    node = ChartRenderer(
        chart_config=chart_config,
        data=sample_dataframe,
        width=400,
        height=300,
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
@pytest.mark.skip(reason="Seaborn histplot bins parameter issue in library code")
async def test_chart_renderer_histplot(context: ProcessingContext, sample_dataframe):
    """Test ChartRenderer with histogram."""
    chart_config = ChartConfig(
        title="Test Histogram",
        data=ChartData(
            series=[
                DataSeries(
                    x="y",
                    plot_type=SeabornPlotType.HISTPLOT,
                )
            ]
        ),
    )
    node = ChartRenderer(
        chart_config=chart_config,
        data=sample_dataframe,
        width=400,
        height=300,
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_chart_renderer_with_hue(context: ProcessingContext, sample_dataframe):
    """Test ChartRenderer with hue grouping."""
    chart_config = ChartConfig(
        title="Scatter with Hue",
        data=ChartData(
            series=[
                DataSeries(
                    x="x",
                    y="y",
                    hue="category",
                    plot_type=SeabornPlotType.SCATTER,
                )
            ]
        ),
    )
    node = ChartRenderer(
        chart_config=chart_config,
        data=sample_dataframe,
        width=400,
        height=300,
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_chart_renderer_custom_size(context: ProcessingContext, sample_dataframe):
    """Test ChartRenderer with custom dimensions."""
    chart_config = ChartConfig(
        data=ChartData(
            series=[
                DataSeries(
                    x="x",
                    y="y",
                    plot_type=SeabornPlotType.SCATTER,
                )
            ]
        ),
    )
    node = ChartRenderer(
        chart_config=chart_config,
        data=sample_dataframe,
        width=800,
        height=600,
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_chart_renderer_with_labels(context: ProcessingContext, sample_dataframe):
    """Test ChartRenderer with axis labels."""
    chart_config = ChartConfig(
        title="Test Chart",
        x_label="X Axis",
        y_label="Y Axis",
        data=ChartData(
            series=[
                DataSeries(
                    x="x",
                    y="y",
                    plot_type=SeabornPlotType.SCATTER,
                )
            ]
        ),
    )
    node = ChartRenderer(
        chart_config=chart_config,
        data=sample_dataframe,
        width=400,
        height=300,
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_chart_renderer_no_despine(context: ProcessingContext, sample_dataframe):
    """Test ChartRenderer without despine."""
    chart_config = ChartConfig(
        data=ChartData(
            series=[
                DataSeries(
                    x="x",
                    y="y",
                    plot_type=SeabornPlotType.SCATTER,
                )
            ]
        ),
    )
    node = ChartRenderer(
        chart_config=chart_config,
        data=sample_dataframe,
        width=400,
        height=300,
        despine=False,
    )
    result = await node.process(context)
    assert isinstance(result, ImageRef)


@pytest.mark.asyncio
async def test_chart_renderer_empty_dataframe_raises_error(context: ProcessingContext):
    """Test ChartRenderer with empty dataframe raises error."""
    chart_config = ChartConfig(
        data=ChartData(
            series=[
                DataSeries(
                    x="x",
                    y="y",
                    plot_type=SeabornPlotType.SCATTER,
                )
            ]
        ),
    )
    # Create an empty DataframeRef
    empty_df = DataframeRef(data=[], columns=[])
    node = ChartRenderer(
        chart_config=chart_config,
        data=empty_df,
        width=400,
        height=300,
    )
    # Empty dataframe should raise a KeyError when trying to access columns
    with pytest.raises(Exception):  # Could be KeyError or other exception
        await node.process(context)


def test_seaborn_enums():
    """Test that seaborn enums have expected values."""
    assert SeabornStyle.DARKGRID == "darkgrid"
    assert SeabornStyle.WHITEGRID == "whitegrid"
    assert SeabornContext.PAPER == "paper"
    assert SeabornContext.NOTEBOOK == "notebook"
    assert SeabornPalette.DEEP == "deep"
    assert SeabornPalette.COLORBLIND == "colorblind"
    assert SeabornFont.SANS_SERIF == "sans-serif"
