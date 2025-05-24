from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class ClusterVisualizationNode(GraphNode):
    """
    Visualize clustering results in 2D space.
    machine learning, visualization, clustering

    Use cases:
    - Cluster analysis
    - Pattern recognition
    - Data distribution visualization
    """

    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Input features (2D data)",
    )
    labels: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Cluster labels",
    )
    centers: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Cluster centers (if available)",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.visualization.ClusterVisualization"


class ConfusionMatrixPlotNode(GraphNode):
    """
    Plot confusion matrix heatmap.
    machine learning, visualization, evaluation, classification

    Use cases:
    - Classification error analysis
    - Model performance visualization
    - Class balance assessment
    """

    y_true: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="True labels",
    )
    y_pred: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Predicted labels",
    )
    normalize: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Whether to normalize the confusion matrix"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.visualization.ConfusionMatrixPlot"


class DecisionBoundaryPlot(GraphNode):
    """
    Visualize classifier decision boundaries in 2D space.
    machine learning, visualization, classification, knn

    Use cases:
    - Decision boundary visualization
    - Model behavior analysis
    - Feature space understanding
    - High-dimensional data visualization through dimension selection
    """

    model: types.SKLearnModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.SKLearnModel(type="sklearn_model", model=None),
        description="Fitted classifier",
    )
    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Training features",
    )
    y: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Training labels",
    )
    mesh_step_size: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.02, description="Step size for creating the mesh grid"
    )
    dim1: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="First dimension index to plot"
    )
    dim2: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1, description="Second dimension index to plot"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.visualization.DecisionBoundaryPlot"


class ElbowCurvePlotNode(GraphNode):
    """
    Plot elbow curve for K-means clustering.
    machine learning, visualization, clustering, model selection

    Use cases:
    - Optimal cluster number selection
    - K-means evaluation
    - Model complexity analysis
    """

    inertias: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Inertia values for different k",
    )
    k_values: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="K values tested",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.visualization.ElbowCurvePlot"


class LearningCurveNode(GraphNode):
    """
    Plot learning curves to diagnose bias/variance.
    machine learning, visualization, evaluation, model selection

    Use cases:
    - Bias-variance diagnosis
    - Sample size impact analysis
    - Model complexity assessment
    """

    model: types.SKLearnModel | GraphNode | tuple[GraphNode, str] = Field(
        default=PydanticUndefined, description="Fitted sklearn model"
    )
    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Training features",
    )
    y: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Training labels",
    )
    cv: int | GraphNode | tuple[GraphNode, str] = Field(
        default=5, description="Number of cross-validation folds"
    )
    n_jobs: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1, description="Number of jobs for parallel processing"
    )
    train_sizes: list[float] | GraphNode | tuple[GraphNode, str] = Field(
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        description="Points on the training learning curve",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.visualization.LearningCurve"


class NMFComponentsPlotNode(GraphNode):
    """
    Visualize NMF components as a heatmap.
    machine learning, visualization, dimensionality reduction, nmf

    Use cases:
    - Inspect learned NMF components
    - Analyze feature patterns
    - Validate decomposition results
    """

    components: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="NMF components matrix (from components_ attribute)",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.visualization.NMFComponentsPlot"


class PlotTSNE(GraphNode):
    """
    Create a t-SNE plot for high-dimensional array data.
    array, tsne, visualization, dimensionality reduction

    Use cases:
    - Visualize clusters in high-dimensional data
    - Explore relationships in complex datasets
    - Reduce dimensionality for data analysis
    """

    array: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description=None,
    )
    color_indices: list[int] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description=None
    )
    perplexity: int | GraphNode | tuple[GraphNode, str] = Field(
        default=30, description=None
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.visualization.PlotTSNE"


class ROCCurveNode(GraphNode):
    """
    Plot Receiver Operating Characteristic (ROC) curve.
    machine learning, visualization, evaluation, classification

    Use cases:
    - Binary classifier evaluation
    - Model comparison
    - Threshold selection
    """

    y_true: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="True binary labels",
    )
    y_score: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Target scores/probabilities",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.visualization.ROCCurve"


class RegressionPlot(GraphNode):
    """
    Create a scatter plot with optional regression line.
    machine learning, visualization, regression, scatter plot

    Use cases:
    - Visualize feature-target relationships
    - Explore linear correlations
    - Show regression line fit
    - Data distribution analysis
    """

    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Feature values (1D)",
    )
    y: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Target values",
    )
    show_regression_line: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Whether to show the regression line"
    )
    x_label: str | GraphNode | tuple[GraphNode, str] = Field(
        default="Feature", description="X-axis label"
    )
    y_label: str | GraphNode | tuple[GraphNode, str] = Field(
        default="Target", description="Y-axis label"
    )
    title: str | GraphNode | tuple[GraphNode, str] = Field(
        default="Regression Plot", description="Plot title"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.visualization.RegressionPlot"


class RegressionPredictionPlotNode(GraphNode):
    """
    Plot actual vs predicted values for regression models.
    machine learning, visualization, evaluation, regression

    Use cases:
    - Regression model evaluation
    - Prediction accuracy visualization
    - Outlier detection
    """

    y_true: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="True values",
    )
    y_pred: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Predicted values",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.visualization.RegressionPredictionPlot"


class RegressionResidualPlotNode(GraphNode):
    """
    Plot residuals for regression analysis.
    machine learning, visualization, evaluation, regression

    Use cases:
    - Model assumptions validation
    - Error pattern detection
    - Heteroscedasticity check
    """

    y_true: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="True values",
    )
    y_pred: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Predicted values",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.visualization.RegressionResidualPlot"
