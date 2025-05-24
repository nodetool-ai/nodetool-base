from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class PredictNode(GraphNode):
    """
    Make predictions using a fitted statsmodels model.
    machine learning, prediction, regression

    Use cases:
    - Making predictions with fitted models
    - Model inference
    - Out-of-sample prediction
    """

    model: types.StatsModelsModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.StatsModelsModel(type="statsmodels_model", model=None),
        description="Fitted statsmodels model",
    )
    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Features to predict on",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.statsmodels.Predict"
