from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class TransformedTargetRegressorNode(GraphNode):
    """
    Meta-estimator to regress on a transformed target.
    machine learning, regression, target transformation

    Use cases:
    - Log-transform regression targets
    - Box-Cox transformations
    - Custom target transformations
    """

    X_train: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Training features",
    )
    y_train: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Training target values",
    )
    regressor: types.SKLearnModel | GraphNode | tuple[GraphNode, str] = Field(
        default=PydanticUndefined, description="Base regressor"
    )
    transformer: types.SKLearnModel | GraphNode | tuple[GraphNode, str] = Field(
        default=PydanticUndefined, description="Target transformer"
    )
    check_inverse: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True,
        description="Whether to check that transform followed by inverse transform gives original targets",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.compose.TransformedTargetRegressor"
