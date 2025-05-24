from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class AccuracyNode(GraphNode):
    """
    Calculate accuracy score for classification.
    machine learning, evaluation, metrics, classification

    Use cases:
    - Model evaluation
    - Classification accuracy assessment
    """

    y_true: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Ground truth labels",
    )
    y_pred: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Predicted labels",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.metrics.Accuracy"


class CalinskiHarabaszNode(GraphNode):
    """
    Calculate Calinski-Harabasz score for clustering.
    machine learning, evaluation, metrics, clustering

    Use cases:
    - Cluster separation assessment
    - Optimal cluster number selection
    """

    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Input samples",
    )
    labels: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Cluster labels",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.metrics.CalinskiHarabasz"


class ConfusionMatrixNode(GraphNode):
    """
    Calculate confusion matrix for classification.
    machine learning, evaluation, metrics, classification

    Use cases:
    - Detailed classification error analysis
    - Model performance visualization
    """

    y_true: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Ground truth labels",
    )
    y_pred: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Predicted labels",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.metrics.ConfusionMatrix"


class DaviesBouldinNode(GraphNode):
    """
    Calculate Davies-Bouldin score for clustering.
    machine learning, evaluation, metrics, clustering

    Use cases:
    - Cluster quality assessment
    - Clustering algorithm comparison
    """

    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Input samples",
    )
    labels: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Cluster labels",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.metrics.DaviesBouldin"


class ExplainedVarianceNode(GraphNode):
    """
    Calculate explained variance score for regression.
    machine learning, evaluation, metrics, regression

    Use cases:
    - Model quality assessment
    - Variance explanation evaluation
    """

    y_true: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Ground truth values",
    )
    y_pred: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Predicted values",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.metrics.ExplainedVariance"


import nodetool.nodes.lib.ml.sklearn.metrics


class F1Node(GraphNode):
    """
    Calculate F1 score for classification.
    machine learning, evaluation, metrics, classification

    Use cases:
    - Model evaluation
    - Balance between precision and recall
    """

    ClassificationMetricsAverage: typing.ClassVar[type] = (
        nodetool.nodes.lib.ml.sklearn.metrics.ClassificationMetricsAverage
    )
    y_true: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Ground truth labels",
    )
    y_pred: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Predicted labels",
    )
    average: nodetool.nodes.lib.ml.sklearn.metrics.ClassificationMetricsAverage = Field(
        default=nodetool.nodes.lib.ml.sklearn.metrics.ClassificationMetricsAverage.BINARY,
        description="Averaging strategy for multiclass: 'binary' (default), 'micro', 'macro', 'weighted'",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.metrics.F1"


class MAENode(GraphNode):
    """
    Calculate Mean Absolute Error for regression.
    machine learning, evaluation, metrics, regression

    Use cases:
    - Model evaluation
    - Average error magnitude assessment
    """

    y_true: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Ground truth values",
    )
    y_pred: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Predicted values",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.metrics.MAE"


class MSENode(GraphNode):
    """
    Calculate Mean Squared Error for regression.
    machine learning, evaluation, metrics, regression

    Use cases:
    - Model evaluation
    - Regression error assessment
    """

    y_true: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Ground truth values",
    )
    y_pred: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Predicted values",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.metrics.MSE"


import nodetool.nodes.lib.ml.sklearn.metrics


class PrecisionNode(GraphNode):
    """
    Calculate precision score for classification.
    machine learning, evaluation, metrics, classification

    Use cases:
    - Model evaluation
    - Precision assessment
    """

    ClassificationMetricsAverage: typing.ClassVar[type] = (
        nodetool.nodes.lib.ml.sklearn.metrics.ClassificationMetricsAverage
    )
    y_true: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Ground truth labels",
    )
    y_pred: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Predicted labels",
    )
    average: nodetool.nodes.lib.ml.sklearn.metrics.ClassificationMetricsAverage = Field(
        default=nodetool.nodes.lib.ml.sklearn.metrics.ClassificationMetricsAverage.BINARY,
        description="Averaging strategy for multiclass: 'binary' (default), 'micro', 'macro', 'weighted'",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.metrics.Precision"


class R2Node(GraphNode):
    """
    Calculate R-squared (coefficient of determination) score for regression.
    machine learning, evaluation, metrics, regression

    Use cases:
    - Model fit assessment
    - Variance explanation evaluation
    """

    y_true: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Ground truth values",
    )
    y_pred: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Predicted values",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.metrics.R2"


class RMSENode(GraphNode):
    """
    Calculate Root Mean Squared Error for regression.
    machine learning, evaluation, metrics, regression

    Use cases:
    - Model evaluation
    - Error magnitude assessment
    """

    y_true: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Ground truth values",
    )
    y_pred: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Predicted values",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.metrics.RMSE"


class ROCCurveNode(GraphNode):
    """
    Calculate ROC curve and AUC score.
    machine learning, evaluation, metrics, classification

    Use cases:
    - Binary classifier evaluation
    - Model comparison
    - Threshold selection
    """

    y_true: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Ground truth labels",
    )
    y_score: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Predicted probabilities or scores",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.metrics.ROCCurve"


import nodetool.nodes.lib.ml.sklearn.metrics


class RecallNode(GraphNode):
    """
    Calculate recall score for classification.
    machine learning, evaluation, metrics, classification

    Use cases:
    - Model evaluation
    - Recall assessment
    """

    ClassificationMetricsAverage: typing.ClassVar[type] = (
        nodetool.nodes.lib.ml.sklearn.metrics.ClassificationMetricsAverage
    )
    y_true: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Ground truth labels",
    )
    y_pred: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Predicted labels",
    )
    average: nodetool.nodes.lib.ml.sklearn.metrics.ClassificationMetricsAverage = Field(
        default=nodetool.nodes.lib.ml.sklearn.metrics.ClassificationMetricsAverage.BINARY,
        description="Averaging strategy for multiclass: 'binary' (default), 'micro', 'macro', 'weighted'",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.metrics.Recall"


class SilhouetteScoreNode(GraphNode):
    """
    Calculate Silhouette score for clustering.
    machine learning, evaluation, metrics, clustering

    Use cases:
    - Cluster quality assessment
    - Clustering algorithm evaluation
    """

    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Input samples",
    )
    labels: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Cluster labels",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.metrics.SilhouetteScore"
