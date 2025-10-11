import numpy as np
from io import BytesIO
from enum import Enum
from pydantic import Field
from matplotlib import pyplot as plt
import seaborn as sns
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import BaseNode
from nodetool.metadata.types import to_numpy
from nodetool.metadata.types import NPArray, ImageRef


class PlotArray(BaseNode):
    """
    Create a plot visualization of array data.
    array, plot, visualization, graph

    Use cases:
    - Visualize trends in array data
    - Create charts for reports or dashboards
    - Debug array outputs in workflows
    """

    class PlotType(str, Enum):
        LINE = "line"
        BAR = "bar"
        SCATTER = "scatter"

    values: NPArray = Field(default=NPArray(), description="Array to plot")
    plot_type: PlotType = Field(
        default=PlotType.LINE, description="Type of plot to create"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arr = to_numpy(self.values)
        sns.set_theme(style="darkgrid")
        if self.plot_type == self.PlotType.LINE:
            plot = sns.lineplot(data=arr)
        elif self.plot_type == self.PlotType.BAR:
            plot = sns.barplot(data=arr)
        elif self.plot_type == self.PlotType.SCATTER:
            plot = sns.scatterplot(data=arr)
        else:
            raise ValueError(f"Invalid plot type: {self.plot_type}")
        fig = plot.get_figure()
        if fig is None:
            raise ValueError("Could not get figure from plot.")
        img_bytes = BytesIO()
        fig.savefig(img_bytes, format="png")
        plt.close(fig)
        return await context.image_from_bytes(img_bytes.getvalue())
