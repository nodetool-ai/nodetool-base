from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class PredictNode(GraphNode):
    """
    Makes predictions using a fitted sklearn model.
    machine learning, prediction, inference

    Use cases:
    - Make predictions on new data
    - Score model performance
    """

    model: types.SKLearnModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.SKLearnModel(type="sklearn_model", model=None),
        description="Fitted sklearn model",
    )
    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Features to predict on",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.Predict"
