from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode

import nodetool.nodes.lib.ml.sklearn.impute


class KNNImputerNode(GraphNode):
    """
    Imputation using k-Nearest Neighbors.
    machine learning, preprocessing, imputation, missing values, knn

    Use cases:
    - Advanced missing value imputation
    - Preserving data relationships
    - Handling multiple missing values
    """

    Weights: typing.ClassVar[type] = (
        nodetool.nodes.lib.ml.sklearn.impute.KNNImputerNode.Weights
    )
    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Input data with missing values",
    )
    n_neighbors: int | GraphNode | tuple[GraphNode, str] = Field(
        default=5, description="Number of neighboring samples to use for imputation"
    )
    weights: nodetool.nodes.lib.ml.sklearn.impute.KNNImputerNode.Weights = Field(
        default=nodetool.nodes.lib.ml.sklearn.impute.KNNImputerNode.Weights.UNIFORM,
        description="Weight function used in prediction: 'uniform' or 'distance'",
    )
    missing_values: str | float | GraphNode | tuple[GraphNode, str] = Field(
        default=nan,
        description="Placeholder for missing values. Can be np.nan or numeric value",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.impute.KNNImputer"


class SimpleImputerNode(GraphNode):
    """
    Imputation transformer for completing missing values.
    machine learning, preprocessing, imputation, missing values

    Use cases:
    - Handling missing values in datasets
    - Basic data cleaning
    - Preparing data for ML models
    """

    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Input data with missing values",
    )
    strategy: str | GraphNode | tuple[GraphNode, str] = Field(
        default="mean",
        description="Imputation strategy: 'mean', 'median', 'most_frequent', or 'constant'",
    )
    fill_value: str | float | GraphNode | tuple[GraphNode, str] = Field(
        default=None,
        description="Value to use when strategy is 'constant'. Can be str or numeric",
    )
    missing_values: str | float | GraphNode | tuple[GraphNode, str] = Field(
        default=nan,
        description="Placeholder for missing values. Can be np.nan or numeric value",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.impute.SimpleImputer"
