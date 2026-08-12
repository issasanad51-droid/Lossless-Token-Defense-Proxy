"""
Structural retrieval layer.

Three retrievers run over the same corpus and are blended by Reciprocal Rank
Fusion:

* :mod:`retrieval.ast_graph`    - syntax-driven index of functions, classes and
                                  the call graph between them
* :mod:`retrieval.vector_store` - vector-space lookup over code-aware terms
* :mod:`retrieval.ppr`          - Personalized PageRank over the call graph
* :mod:`retrieval.rrf`          - rank fusion + overlap grouping + density ranking
* :mod:`retrieval.hybrid`       - runs the retrievers concurrently and blends them
"""

from __future__ import annotations

from retrieval.ast_graph import CodeGraph, CodeSpan, EdgeKind, build_graph
from retrieval.hybrid import HybridRetriever, RetrievalResult
from retrieval.ppr import personalized_pagerank, ppr_rank
from retrieval.rrf import CodeBlock, RankedList, reciprocal_rank_fusion
from retrieval.vector_store import VectorStore, code_tokenize

__all__ = [
    "CodeGraph",
    "CodeSpan",
    "EdgeKind",
    "build_graph",
    "VectorStore",
    "code_tokenize",
    "personalized_pagerank",
    "ppr_rank",
    "reciprocal_rank_fusion",
    "RankedList",
    "CodeBlock",
    "HybridRetriever",
    "RetrievalResult",
]
