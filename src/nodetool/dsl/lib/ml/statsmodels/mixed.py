from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class MixedLMNode(GraphNode):
    """
    Linear Mixed Effects Model.
    statistics, regression, mixed effects, hierarchical model

    Use cases:
    - Hierarchical/nested data
    - Repeated measures analysis
    - Longitudinal data analysis
    - Clustered data
    """

    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Features/independent variables",
    )
    y: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Target/dependent variable",
    )
    groups: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Group labels for random effects",
    )
    use_reml: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Use REML estimation"
    )
    maxiter: int | GraphNode | tuple[GraphNode, str] = Field(
        default=50, description="Maximum number of iterations"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.statsmodels.mixed.MixedLM"


class MixedLMPredictNode(GraphNode):
    """
    Make predictions using a fitted Mixed Linear Model.
    statistics, regression, prediction, mixed effects

    Use cases:
    - Prediction with mixed effects models
    - Out-of-sample prediction
    - Model evaluation
    """

    model: types.StatsModelsModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.StatsModelsModel(type="statsmodels_model", model=None),
        description="Fitted Mixed LM model",
    )
    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Features for prediction",
    )
    groups: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Group labels for prediction",
    )
    confidence_level: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.95,
        description="Confidence level for prediction intervals (between 0 and 1)",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.statsmodels.mixed.MixedLMPredict"
