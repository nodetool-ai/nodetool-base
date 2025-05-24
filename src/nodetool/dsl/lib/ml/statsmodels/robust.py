from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode

import nodetool.nodes.lib.ml.statsmodels.robust


class RLMNode(GraphNode):
    """
    Robust Linear Model Regression.
    statistics, regression, robust, outliers

    Use cases:
    - Regression with outliers
    - Robust parameter estimation
    - Non-normal error distributions
    """

    MEstimator: typing.ClassVar[type] = (
        nodetool.nodes.lib.ml.statsmodels.robust.MEstimator
    )
    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Features/independent variables",
    )
    y: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Target/dependent variable",
    )
    M: nodetool.nodes.lib.ml.statsmodels.robust.MEstimator = Field(
        default=nodetool.nodes.lib.ml.statsmodels.robust.MEstimator.HUBER,
        description="M-estimator ('huber', 'bisquare', etc.)",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.statsmodels.robust.RLM"
