"""
FAISS nodes for Nodetool.
"""

from enum import Enum
from typing import TYPE_CHECKING, Any, TypedDict

from nodetool.metadata.types import FaissIndex, NPArray
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field

if TYPE_CHECKING:
    import faiss
    import numpy as np


class FaissNode(BaseNode):
    """Base class for FAISS nodes.

    vector, faiss, index, search
    """

    @classmethod
    def is_cacheable(cls):
        return False

    @classmethod
    def is_visible(cls):
        return cls is not FaissNode


def _ensure_2d_float32(array: "np.ndarray") -> "np.ndarray":
    import numpy as np

    if array.ndim == 1:
        array = array.reshape(1, -1)
    return np.ascontiguousarray(array.astype(np.float32))


def _ensure_1d_int64(array: "np.ndarray") -> "np.ndarray":
    import numpy as np

    array = np.ascontiguousarray(array.astype(np.int64))
    return array.reshape(-1)


class Metric(str, Enum):
    L2 = "L2"
    IP = "IP"


class CreateIndexFlatL2(FaissNode):
    """
    Create a FAISS IndexFlatL2.
    faiss, index, l2, create
    """

    dim: int = Field(default=768, ge=1, description="Embedding dimensionality")

    async def process(self, context: ProcessingContext) -> FaissIndex:
        import faiss

        idx = faiss.IndexFlatL2(self.dim)
        return FaissIndex(index=idx)


class CreateIndexFlatIP(FaissNode):
    """
    Create a FAISS IndexFlatIP (inner product / cosine with normalized vectors).
    faiss, index, ip, create
    """

    dim: int = Field(default=768, ge=1, description="Embedding dimensionality")

    async def process(self, context: ProcessingContext) -> FaissIndex:
        import faiss

        idx = faiss.IndexFlatIP(self.dim)
        return FaissIndex(index=idx)


class CreateIndexIVFFlat(FaissNode):
    """
    Create a FAISS IndexIVFFlat (inverted file index with flat quantizer).
    faiss, index, ivf, create
    """

    dim: int = Field(default=768, ge=1, description="Embedding dimensionality")
    nlist: int = Field(default=1024, ge=1, description="Number of Voronoi cells")
    metric: Metric = Field(default=Metric.L2, description="Distance metric")

    async def process(self, context: ProcessingContext) -> FaissIndex:
        import faiss

        if self.metric == Metric.L2:
            quantizer = faiss.IndexFlatL2(self.dim)
            metric_const = faiss.METRIC_L2
        else:
            quantizer = faiss.IndexFlatIP(self.dim)
            metric_const = faiss.METRIC_INNER_PRODUCT

        idx = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, metric_const)
        return FaissIndex(index=idx)


class TrainIndex(FaissNode):
    """
    Train a FAISS index with training vectors (required for IVF indices).
    faiss, train, index
    """

    index: FaissIndex = Field(default=FaissIndex(), description="FAISS index")
    vectors: NPArray = Field(default=NPArray(), description="Training vectors (n, d)")

    async def process(self, context: ProcessingContext) -> FaissIndex:
        if self.index.index is None:
            raise ValueError("FAISS index is not set")
        if self.vectors.is_empty():
            raise ValueError("Training vectors are empty")

        idx = self.index.index
        x = _ensure_2d_float32(self.vectors.to_numpy())
        if hasattr(idx, "d") and x.shape[1] != idx.d:
            raise ValueError(
                f"Vector dimension {x.shape[1]} does not match index.d {idx.d}"
            )

        if hasattr(idx, "is_trained") and not idx.is_trained:
            idx.train(x)
        return self.index


class AddVectors(FaissNode):
    """
    Add vectors to a FAISS index.
    faiss, add, vectors
    """

    index: FaissIndex = Field(default=FaissIndex(), description="FAISS index")
    vectors: NPArray = Field(default=NPArray(), description="Vectors to add (n, d)")

    async def process(self, context: ProcessingContext) -> FaissIndex:
        if self.index.index is None:
            raise ValueError("FAISS index is not set")
        if self.vectors.is_empty():
            raise ValueError("Vectors are empty")

        idx = self.index.index
        x = _ensure_2d_float32(self.vectors.to_numpy())
        if hasattr(idx, "d") and x.shape[1] != idx.d:
            raise ValueError(
                f"Vector dimension {x.shape[1]} does not match index.d {idx.d}"
            )

        if hasattr(idx, "is_trained") and not idx.is_trained:
            raise ValueError("Index must be trained before adding vectors")

        idx.add(x)
        return self.index


class AddWithIds(FaissNode):
    """
    Add vectors with explicit integer IDs to a FAISS index.
    faiss, add, ids, vectors
    """

    index: FaissIndex = Field(default=FaissIndex(), description="FAISS index")
    vectors: NPArray = Field(default=NPArray(), description="Vectors to add (n, d)")
    ids: NPArray = Field(default=NPArray(), description="1-D int64 IDs (n,)")

    async def process(self, context: ProcessingContext) -> FaissIndex:
        import faiss
        import numpy as np

        if self.index.index is None:
            raise ValueError("FAISS index is not set")
        if self.vectors.is_empty():
            raise ValueError("Vectors are empty")
        if self.ids.is_empty():
            raise ValueError("IDs are empty")

        idx = self.index.index
        x = _ensure_2d_float32(self.vectors.to_numpy())
        labels = _ensure_1d_int64(self.ids.to_numpy())

        if hasattr(idx, "d") and x.shape[1] != idx.d:
            raise ValueError(
                f"Vector dimension {x.shape[1]} does not match index.d {idx.d}"
            )
        if x.shape[0] != labels.shape[0]:
            raise ValueError("Vectors and IDs must have the same length")

        if hasattr(idx, "is_trained") and not idx.is_trained:
            raise ValueError("Index must be trained before adding vectors")

        # Always ensure we have an ID-mapped index for robust ID support
        if not isinstance(
            idx, (faiss.IndexIDMap, getattr(faiss, "IndexIDMap2", tuple()))
        ):
            idmap_cls = getattr(faiss, "IndexIDMap2", faiss.IndexIDMap)
            idmap = idmap_cls(idx)
            self.index.index = idmap
            idx = idmap

        idx.add_with_ids(x, labels)  # type: ignore
        return self.index


class Search(FaissNode):
    """
    Search a FAISS index with query vectors, returning distances and indices.
    faiss, search, query, knn
    """

    index: FaissIndex = Field(default=FaissIndex(), description="FAISS index")
    query: NPArray = Field(
        default=NPArray(), description="Query vectors (m, d) or (d,)"
    )
    k: int = Field(default=5, ge=1, description="Number of nearest neighbors")
    nprobe: int | None = Field(default=None, description="nprobe for IVF indices")

    class OutputType(TypedDict):
        distances: NPArray
        indices: NPArray

    async def process(self, context: ProcessingContext) -> OutputType:
        import numpy as np

        if self.index.index is None:
            raise ValueError("FAISS index is not set")
        if self.query.is_empty():
            raise ValueError("Query vectors are empty")

        idx = self.index.index
        if self.nprobe is not None and hasattr(idx, "nprobe"):
            idx.nprobe = int(self.nprobe)

        q = _ensure_2d_float32(self.query.to_numpy())
        if hasattr(idx, "d") and q.shape[1] != idx.d:
            raise ValueError(
                f"Query dimension {q.shape[1]} does not match index.d {idx.d}"
            )

        distances, indices = idx.search(q, self.k)
        return {
            "distances": NPArray.from_numpy(
                np.ascontiguousarray(distances.astype(np.float32))
            ),
            "indices": NPArray.from_numpy(
                np.ascontiguousarray(indices.astype(np.int64))
            ),
        }
