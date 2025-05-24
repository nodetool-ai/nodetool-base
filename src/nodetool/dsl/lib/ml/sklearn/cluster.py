from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode

import nodetool.nodes.lib.ml.sklearn.cluster


class AgglomerativeClusteringNode(GraphNode):
    """
    Hierarchical clustering using a bottom-up approach.
    machine learning, clustering, unsupervised, hierarchical

    Use cases:
    - Hierarchical data organization
    - Taxonomy creation
    - Document hierarchies
    """

    AgglomerativeClusteringLinkage: typing.ClassVar[type] = (
        nodetool.nodes.lib.ml.sklearn.cluster.AgglomerativeClusteringLinkage
    )
    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Features for clustering",
    )
    n_clusters: int | GraphNode | tuple[GraphNode, str] = Field(
        default=2, description="Number of clusters"
    )
    linkage: nodetool.nodes.lib.ml.sklearn.cluster.AgglomerativeClusteringLinkage = (
        Field(
            default=nodetool.nodes.lib.ml.sklearn.cluster.AgglomerativeClusteringLinkage.WARD,
            description="Linkage criterion: 'ward', 'complete', 'average', 'single'",
        )
    )
    metric: str | GraphNode | tuple[GraphNode, str] = Field(
        default="euclidean", description="Metric used for distance computation"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.cluster.AgglomerativeClustering"


class DBSCANNode(GraphNode):
    """
    Density-Based Spatial Clustering of Applications with Noise.
    machine learning, clustering, unsupervised, density-based

    Use cases:
    - Anomaly detection
    - Spatial clustering
    - Finding clusters of arbitrary shape
    """

    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Features for clustering",
    )
    eps: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.5, description="Maximum distance between samples for neighborhood"
    )
    min_samples: int | GraphNode | tuple[GraphNode, str] = Field(
        default=5, description="Minimum number of samples in a neighborhood"
    )
    metric: str | GraphNode | tuple[GraphNode, str] = Field(
        default="euclidean", description="Metric to compute distances"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.cluster.DBSCAN"


class KMeansNode(GraphNode):
    """
    K-Means clustering algorithm.
    machine learning, clustering, unsupervised

    Use cases:
    - Customer segmentation
    - Image compression
    - Document clustering
    """

    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Features for clustering",
    )
    n_clusters: int | GraphNode | tuple[GraphNode, str] = Field(
        default=8, description="Number of clusters"
    )
    random_state: int | GraphNode | tuple[GraphNode, str] = Field(
        default=42, description="Random state for reproducibility"
    )
    max_iter: int | GraphNode | tuple[GraphNode, str] = Field(
        default=300, description="Maximum number of iterations"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.ml.sklearn.cluster.KMeans"
