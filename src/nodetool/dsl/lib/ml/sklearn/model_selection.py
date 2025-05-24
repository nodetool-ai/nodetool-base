from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class GridSearchNode(GraphNode):
    """
    Exhaustive search over specified parameter values.
    machine learning, hyperparameter tuning, model selection

    Use cases:
    - Hyperparameter optimization
    - Model selection
    - Automated model tuning
    """

    model: types.SKLearnModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.SKLearnModel(type="sklearn_model", model=None),
        description="Base sklearn model",
    )
    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Training features",
    )
    y: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Training target values",
    )
    param_grid: dict[str, list[Any]] | GraphNode | tuple[GraphNode, str] = Field(
        default={},
        description="Dictionary with parameters names (string) as keys and lists of parameter settings to try",
    )
    cv: int | GraphNode | tuple[GraphNode, str] = Field(
        default=5, description="Number of folds for cross-validation"
    )
    scoring: str | GraphNode | tuple[GraphNode, str] = Field(
        default="accuracy", description="Scoring metric to use for evaluation"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.model_selection.GridSearch"


class KFoldCrossValidationNode(GraphNode):
    """
    K-Fold Cross Validation for model evaluation.
    machine learning, model evaluation, cross validation

    Use cases:
    - Model performance estimation
    - Hyperparameter tuning
    - Assessing model stability
    """

    model: types.SKLearnModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.SKLearnModel(type="sklearn_model", model=None),
        description="Sklearn model to evaluate",
    )
    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Features for cross validation",
    )
    y: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Target values",
    )
    n_splits: int | GraphNode | tuple[GraphNode, str] = Field(
        default=5, description="Number of folds"
    )
    shuffle: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Whether to shuffle the data"
    )
    random_state: int | GraphNode | tuple[GraphNode, str] = Field(
        default=42, description="Random state for reproducibility"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.model_selection.KFoldCrossValidation"


class TrainTestSplitNode(GraphNode):
    """
    Split arrays into random train and test subsets.
    machine learning, data splitting, model evaluation

    Use cases:
    - Preparing data for model training
    - Model evaluation
    - Preventing data leakage
    """

    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Features to split",
    )
    y: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Target values to split",
    )
    test_size: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.25,
        description="Proportion of the dataset to include in the test split",
    )
    random_state: int | GraphNode | tuple[GraphNode, str] = Field(
        default=42, description="Random state for reproducibility"
    )
    shuffle: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Whether to shuffle the data"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.model_selection.TrainTestSplit"
