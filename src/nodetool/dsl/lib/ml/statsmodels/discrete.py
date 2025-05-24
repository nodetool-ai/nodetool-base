from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class LogitNode(GraphNode):
    """
    Logistic Regression using statsmodels.
    statistics, regression, classification, logistic

    Use cases:
    - Binary classification
    - Probability estimation
    - Statistical inference for classification
    """

    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Features/independent variables",
    )
    y: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Binary target variable (0/1)",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.statsmodels.discrete.Logit"


class MultinomialLogitNode(GraphNode):
    """
    Multinomial Logistic Regression for nominal outcomes.
    statistics, regression, multinomial, classification

    Use cases:
    - Multiple category classification
    - Nominal categorical outcomes
    - Choice modeling
    """

    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Features/independent variables",
    )
    y: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Categorical target variable",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.statsmodels.discrete.MultinomialLogit"


class NegativeBinomialNode(GraphNode):
    """
    Negative Binomial Regression for overdispersed count data.
    statistics, regression, count-data, negative-binomial

    Use cases:
    - Overdispersed count data
    - When variance exceeds mean
    - More flexible than Poisson
    """

    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Features/independent variables",
    )
    y: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Count data target variable",
    )
    exposure: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Optional exposure variable",
    )
    offset: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Optional offset term",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.statsmodels.discrete.NegativeBinomial"


class PoissonNode(GraphNode):
    """
    Poisson Regression for count data.
    statistics, regression, count-data, poisson

    Use cases:
    - Modeling count data
    - Rate data analysis
    - Event frequency prediction
    """

    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Features/independent variables",
    )
    y: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Count data target variable",
    )
    exposure: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Optional exposure variable",
    )
    offset: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Optional offset term",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.statsmodels.discrete.Poisson"
