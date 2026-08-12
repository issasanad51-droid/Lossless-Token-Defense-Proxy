"""
Hybrid retriever: three strategies in parallel, blended by RRF.

    vector lookup  ─┐
    AST structure  ─┼─> Reciprocal Rank Fusion ─> overlap grouping ─> top-3 blocks
    PPR            ─┘

Concurrency uses a thread pool. The honest justification: the AST/PPR/lexical
work here is CPU-bound Python, so threads overlap rather than truly parallelize
it. The reason to keep them is that the *real* deployment swaps the vector stage
for a network call to an embedding API or vector DB, and that stage is
I/O-bound - exactly what threads are for. The pool also isolates failures: one
retriever raising does not lose the other two.

PPR needs seed nodes. They come from the vector stage, so the structural walk
starts wherever the query actually landed - that is what makes the PageRank
*personalized* rather than a static global importance score.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from retrieval.ast_graph import CodeGraph, CodeSpan, EdgeKind, build_graph
from retrieval.ppr import ppr_rank
from retrieval.rrf import CodeBlock, RankedList, reciprocal_rank_fusion
from retrieval.vector_store import VectorStore, code_tokenize

logger = logging.getLogger(__name__)

__all__ = ["HybridRetriever", "RetrievalResult"]


@dataclass
class RetrievalResult:
    """Everything the caller needs, including why each block was chosen."""

    query: str
    blocks: List[CodeBlock]
    rankings: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def total_lines(self) -> int:
        return sum(b.line_count for b in self.blocks)

    def to_context(self, sources: Mapping[str, str], *, header: bool = True) -> str:
        """Render the blocks as a compact context string for the model."""
        chunks: List[str] = []
        for block in self.blocks:
            body = block.text(sources)
            if not body:
                continue
            if header:
                label = block.primary or "block"
                chunks.append(f"# {block.path}:{block.start_line}-{block.end_line} ({label})\n{body}")
            else:
                chunks.append(body)
        return "\n\n".join(chunks)

    def explain(self) -> str:
        lines = [f'query: "{self.query}"']
        for i, b in enumerate(self.blocks, 1):
            agree = ", ".join(f"{r}#{p}" for r, p in sorted(b.contributors.items()))
            lines.append(
                f"  {i}. {b.path}:{b.start_line}-{b.end_line} "
                f"[{b.primary or '?'}] lines={b.line_count} "
                f"score={b.fused_score:.5f} density={b.density:.6f} via {agree or 'n/a'}"
            )
        if self.errors:
            for name, err in self.errors.items():
                lines.append(f"  !! {name} failed: {err}")
        timing = " ".join(f"{k}={v:.1f}ms" for k, v in self.timings_ms.items())
        if timing:
            lines.append(f"  timings: {timing}")
        return "\n".join(lines)


class HybridRetriever:
    """Vector + AST + PPR retrieval fused with RRF."""

    def __init__(
        self,
        sources: Mapping[str, str],
        *,
        embedder: Optional[Callable[[str], Sequence[float]]] = None,
        weights: Optional[Mapping[str, float]] = None,
        rrf_k: int = 60,
        max_workers: int = 3,
    ) -> None:
        self.sources = dict(sources)
        self.graph: CodeGraph = build_graph(self.sources)
        self.store = VectorStore(embedder=embedder)
        self.rrf_k = rrf_k
        self.max_workers = max_workers
        self.weights: Dict[str, float] = dict(weights or {"vector": 1.0, "ast": 0.8, "ppr": 1.0})

        for node_id, span in self.graph.spans.items():
            text = span.text_for_index()
            if text.strip():
                self.store.add(node_id, text)

    # -- individual retrievers ---------------------------------------------

    def _vector_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        return self.store.search(query, top_k=top_k)

    def _ast_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Structural/symbolic match on names, signatures and decorators.

        Complements the vector stage: this is precision-oriented (exact
        identifier hits) where TF-IDF is recall-oriented.
        """
        terms = set(code_tokenize(query))
        if not terms:
            return []

        scored: List[Tuple[str, float]] = []
        for node_id, span in self.graph.spans.items():
            name_terms = set(code_tokenize(span.name))
            qual_terms = set(code_tokenize(span.qualname))
            sig_terms = set(code_tokenize(span.signature))
            dec_terms = set(code_tokenize(" ".join(span.decorators)))
            doc_terms = set(code_tokenize(span.docstring))

            score = 0.0
            score += 3.0 * len(terms & name_terms)
            score += 1.5 * len(terms & (qual_terms - name_terms))
            score += 1.0 * len(terms & (sig_terms - name_terms))
            score += 0.8 * len(terms & dec_terms)
            score += 0.5 * len(terms & doc_terms)

            # Exact full-name match is the strongest possible structural signal.
            if span.name.lower() in {t.lower() for t in terms}:
                score += 4.0
            # Prefer callable definitions over whole modules.
            if span.kind in ("function", "method"):
                score *= 1.15
            elif span.kind == "module":
                score *= 0.6

            if score > 0:
                scored.append((node_id, score))

        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored[:top_k]

    def _ppr_search(self, seeds: Mapping[str, float], top_k: int) -> List[Tuple[str, float]]:
        if not seeds:
            return []
        return ppr_rank(self.graph, seeds, top_k=top_k)

    def seed_nodes(self, query: str, *, n_seeds: int = 5) -> Dict[str, float]:
        """Pick PPR seeds by blending the two content-based retrievers."""
        combined: Dict[str, float] = {}
        for node_id, score in self._vector_search(query, top_k=n_seeds):
            combined[node_id] = combined.get(node_id, 0.0) + score
        for node_id, score in self._ast_search(query, top_k=n_seeds):
            combined[node_id] = combined.get(node_id, 0.0) + score / 10.0

        if not combined:
            return {}
        top = sorted(combined.items(), key=lambda kv: (-kv[1], kv[0]))[:n_seeds]
        total = sum(v for _, v in top) or 1.0
        return {nid: v / total for nid, v in top}

    # -- the pipeline -------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        top_n: int = 3,
        candidates_per_retriever: int = 20,
        merge_gap: int = 2,
        rank_by: str = "density",
        max_block_lines: Optional[int] = None,
        seed_nodes: Optional[Mapping[str, float]] = None,
    ) -> RetrievalResult:
        """Run all three retrievers concurrently and fuse them."""
        timings: Dict[str, float] = {}
        errors: Dict[str, str] = {}
        rankings: Dict[str, List[Tuple[str, float]]] = {}

        # PPR is personalized on where the content retrievers landed.
        t0 = time.perf_counter()
        seeds = dict(seed_nodes) if seed_nodes is not None else self.seed_nodes(query)
        timings["seeding"] = (time.perf_counter() - t0) * 1000

        jobs: Dict[str, Callable[[], List[Tuple[str, float]]]] = {
            "vector": lambda: self._vector_search(query, candidates_per_retriever),
            "ast": lambda: self._ast_search(query, candidates_per_retriever),
            "ppr": lambda: self._ppr_search(seeds, candidates_per_retriever),
        }

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}
            starts = {}
            for name, fn in jobs.items():
                starts[name] = time.perf_counter()
                futures[pool.submit(fn)] = name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    rankings[name] = future.result()
                except Exception as exc:                      # one failure != total failure
                    logger.warning("retriever %s failed: %s", name, exc)
                    errors[name] = f"{type(exc).__name__}: {exc}"
                    rankings[name] = []
                timings[name] = (time.perf_counter() - starts[name]) * 1000

        t0 = time.perf_counter()
        ranked_lists = [
            RankedList(name=name, ranking=rankings.get(name, []), weight=self.weights.get(name, 1.0))
            for name in ("vector", "ast", "ppr")
        ]
        blocks = reciprocal_rank_fusion(
            ranked_lists,
            self.graph,
            k=self.rrf_k,
            top_n=top_n,
            merge_gap=merge_gap,
            rank_by=rank_by,
            max_block_lines=max_block_lines,
        )
        timings["fusion"] = (time.perf_counter() - t0) * 1000

        return RetrievalResult(
            query=query,
            blocks=blocks,
            rankings=rankings,
            timings_ms=timings,
            errors=errors,
        )

    def stats(self) -> Dict[str, int]:
        return {**self.graph.stats(), "indexed_docs": len(self.store)}
