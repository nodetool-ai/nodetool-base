from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode

import nodetool.nodes.lib.ml.statsmodels.glm
import nodetool.nodes.lib.ml.statsmodels.glm


class GLMNode(GraphNode):
    """
    Generalized Linear Models using statsmodels.
    machine learning, regression, generalized linear models

    Use cases:
    - Various types of regression (linear, logistic, poisson, etc.)
    - Handling non-normal error distributions
    - Complex regression analysis
    """

    GLMFamily: typing.ClassVar[type] = nodetool.nodes.lib.ml.statsmodels.glm.GLMFamily
    GLMLink: typing.ClassVar[type] = nodetool.nodes.lib.ml.statsmodels.glm.GLMLink
    X_train: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Training features",
    )
    y_train: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Training target values",
    )
    family: nodetool.nodes.lib.ml.statsmodels.glm.GLMFamily = Field(
        default=nodetool.nodes.lib.ml.statsmodels.glm.GLMFamily.GAUSSIAN,
        description="Error distribution family",
    )
    link: nodetool.nodes.lib.ml.statsmodels.glm.GLMLink = Field(
        default=nodetool.nodes.lib.ml.statsmodels.glm.GLMLink.IDENTITY,
        description="Link function",
    )
    alpha: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.0, description="L2 regularization parameter"
    )
    max_iter: int | GraphNode | tuple[GraphNode, str] = Field(
        default=100, description="Maximum number of iterations"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.statsmodels.glm.GLM"


class GLMPredictNode(GraphNode):
    """
    Make predictions using a fitted GLM model.
    machine learning, regression, prediction, generalized linear models

    Use cases:
    - Prediction with GLM models
    - Out-of-sample prediction
    - Model evaluation
    """

    model: types.SKLearnModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.SKLearnModel(type="sklearn_model", model=None),
        description="Fitted GLM model",
    )
    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Features to predict on",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.statsmodels.glm.GLMPredict"
