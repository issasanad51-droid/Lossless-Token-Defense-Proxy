"""
Advanced Reciprocal Rank Fusion.

Standard RRF scores a document as ``sum over rankers of 1 / (k + rank)``. It
needs no score calibration between rankers, which is exactly why it suits a
hybrid of cosine similarity (0..1), PageRank (a probability) and lexical
overlap - three scales that are meaningless to compare directly.

This module adds the part that plain RRF gets wrong for code:

**Overlap grouping.** A class, its method and an enclosing block are three
separate nodes covering the *same lines*. Naive fusion returns all three and
burns the context window re-sending nested copies of one region. Spans that
overlap or sit within ``merge_gap`` lines of each other in the same file are
merged into a single :class:`CodeBlock` whose line range is the union.

**Density ranking.** Merged blocks are then ranked by *density* - fused
relevance per line - so a tight 12-line function that three retrievers agreed on
outranks a 400-line module that merely contains it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from retrieval.ast_graph import CodeGraph, CodeSpan

__all__ = ["RankedList", "CodeBlock", "reciprocal_rank_fusion", "fuse_ranks"]


@dataclass
class RankedList:
    """One retriever's output."""

    name: str
    ranking: Sequence[Tuple[str, float]]     # ordered [(node_id, raw_score)]
    weight: float = 1.0


# Below this many lines a block is not meaningfully "cheaper" - charging a
# 2-line property the same floor as a 12-line function stops trivia from winning
# the density race on a technicality.
MIN_EFFECTIVE_LINES = 12

# Sublinear length penalty (cf. BM25 length normalization). 1.0 would be pure
# score-per-line, which over-rewards fragments; 0.0 ignores size entirely.
LENGTH_PENALTY = 0.35


@dataclass
class CodeBlock:
    """A merged, contiguous region of one file assembled from ranked spans."""

    path: str
    start_line: int
    end_line: int
    node_ids: List[str] = field(default_factory=list)
    fused_score: float = 0.0
    contributors: Dict[str, int] = field(default_factory=dict)   # retriever -> best rank
    primary: Optional[str] = None                                 # human label for the block
    symbols: List[str] = field(default_factory=list)              # every symbol inside, in order

    @property
    def line_count(self) -> int:
        return max(1, self.end_line - self.start_line + 1)

    @property
    def density(self) -> float:
        """Fused relevance per unit of context cost.

        Not a naive ``score / lines``: that ranks a 2-line property found once
        above a 30-line function that every retriever agreed on. The length
        penalty is sublinear and floored, so size still matters but cannot be
        gamed by returning fragments.
        """
        effective = max(self.line_count, MIN_EFFECTIVE_LINES)
        return self.fused_score / (effective ** LENGTH_PENALTY)

    @property
    def agreement(self) -> int:
        """How many independent retrievers surfaced this block."""
        return len(self.contributors)

    def text(self, sources: Mapping[str, str]) -> str:
        src = sources.get(self.path)
        if src is None:
            return ""
        lines = src.splitlines()
        return "\n".join(lines[self.start_line - 1 : self.end_line])

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"CodeBlock({self.path}:{self.start_line}-{self.end_line}, "
            f"score={self.fused_score:.5f}, density={self.density:.6f}, "
            f"agree={self.agreement})"
        )


def fuse_ranks(
    ranked_lists: Sequence[RankedList],
    *,
    k: int = 60,
) -> Dict[str, Tuple[float, Dict[str, int]]]:
    """Plain RRF. Returns ``{node_id: (fused_score, {retriever: rank})}``.

    ``k`` (default 60, per Cormack et al. 2009) damps the influence of the very
    top ranks so one confident retriever cannot dominate the blend.
    """
    fused: Dict[str, float] = {}
    provenance: Dict[str, Dict[str, int]] = {}

    for rl in ranked_lists:
        if rl.weight <= 0:
            continue
        for position, (node_id, _score) in enumerate(rl.ranking, start=1):
            fused[node_id] = fused.get(node_id, 0.0) + rl.weight / (k + position)
            provenance.setdefault(node_id, {})[rl.name] = position

    return {nid: (score, provenance.get(nid, {})) for nid, score in fused.items()}


def _drop_enclosing(
    candidates: List[Tuple[str, float, Dict[str, int], CodeSpan]],
) -> List[Tuple[str, float, Dict[str, int], CodeSpan]]:
    """Remove spans that merely *contain* other, more specific candidates.

    A module span covers its whole file, so leaving it in makes the merge step
    collapse every hit in that file into one giant block - the exact context
    bloat this layer exists to prevent. Its relevance is not discarded: the
    score is folded into the specific children it encloses, which is what the
    enclosing match was really evidence for.
    """
    if len(candidates) < 2:
        return candidates

    # Pass 1: classify. Decided against the *original* scores so the outcome
    # does not depend on iteration order.
    children_of: Dict[int, List[int]] = {}
    for i, (_, _, _, span) in enumerate(candidates):
        children_of[i] = [
            j
            for j, (_, _, _, other) in enumerate(candidates)
            if j != i
            and other.path == span.path
            and other.line_count < span.line_count
            and other.start_line >= span.start_line
            and other.end_line <= span.end_line
        ]

    # Pass 2: accumulate inherited score/provenance from every dropped parent.
    bonus: Dict[int, float] = {}
    inherited: Dict[int, Dict[str, int]] = {}
    for i, children in children_of.items():
        if not children:
            continue
        _, score, prov, _ = candidates[i]
        share = score / len(children)
        for j in children:
            bonus[j] = bonus.get(j, 0.0) + share
            slot = inherited.setdefault(j, {})
            for retriever, rank in prov.items():
                best = slot.get(retriever)
                slot[retriever] = rank if best is None else min(best, rank)

    # Pass 3: emit only the leaf-most spans, enriched.
    kept: List[Tuple[str, float, Dict[str, int], CodeSpan]] = []
    for i, (nid, score, prov, span) in enumerate(candidates):
        if children_of[i]:
            continue
        merged = dict(prov)
        for retriever, rank in inherited.get(i, {}).items():
            best = merged.get(retriever)
            merged[retriever] = rank if best is None else min(best, rank)
        kept.append((nid, score + bonus.get(i, 0.0), merged, span))

    return kept


def _merge_spans(
    candidates: List[Tuple[str, float, Dict[str, int], CodeSpan]],
    merge_gap: int,
) -> List[CodeBlock]:
    """Union overlapping / near-adjacent spans per file into blocks."""
    by_path: Dict[str, List[Tuple[str, float, Dict[str, int], CodeSpan]]] = {}
    for item in candidates:
        by_path.setdefault(item[3].path, []).append(item)

    blocks: List[CodeBlock] = []

    for path, items in by_path.items():
        # Sweep line: sort by start, extend the open block while it touches.
        items.sort(key=lambda it: (it[3].start_line, it[3].end_line))
        current: Optional[CodeBlock] = None

        for node_id, score, prov, span in items:
            if current is not None and span.start_line <= current.end_line + merge_gap:
                current.start_line = min(current.start_line, span.start_line)
                current.end_line = max(current.end_line, span.end_line)
                current.node_ids.append(node_id)
                current.fused_score += score
                for retriever, rank in prov.items():
                    best = current.contributors.get(retriever)
                    current.contributors[retriever] = rank if best is None else min(best, rank)
                continue

            if current is not None:
                blocks.append(current)
            current = CodeBlock(
                path=path,
                start_line=span.start_line,
                end_line=span.end_line,
                node_ids=[node_id],
                fused_score=score,
                contributors=dict(prov),
            )

        if current is not None:
            blocks.append(current)

    return blocks


def _choose_primary(block: CodeBlock, graph: CodeGraph) -> None:
    """Label the block by the symbols it actually contains.

    A merged block can span several sibling definitions, so labelling it with
    just the smallest one is misleading - the text the caller sees would start
    at a different symbol than the label names. Order by position and name the
    first, noting how many others came along.
    """
    spans = [
        graph.spans[nid]
        for nid in block.node_ids
        if nid in graph.spans and graph.spans[nid].kind != "module"
    ]
    if not spans:
        spans = [graph.spans[nid] for nid in block.node_ids if nid in graph.spans]
    if not spans:
        block.primary = None
        return

    spans.sort(key=lambda s: (s.start_line, s.line_count))
    block.symbols = [s.qualname for s in spans]
    head = spans[0].qualname
    extra = len(spans) - 1
    block.primary = f"{head} +{extra} more" if extra > 0 else head


def reciprocal_rank_fusion(
    ranked_lists: Sequence[RankedList],
    graph: CodeGraph,
    *,
    k: int = 60,
    top_n: int = 3,
    merge_gap: int = 2,
    rank_by: str = "density",
    max_block_lines: Optional[int] = None,
    agreement_bonus: float = 0.15,
    drop_enclosing: bool = True,
) -> List[CodeBlock]:
    """Fuse rankings, group overlapping spans, return the ``top_n`` densest blocks.

    Parameters
    ----------
    merge_gap:
        Spans separated by at most this many lines are treated as one region.
        ``0`` merges only true overlaps.
    rank_by:
        ``density`` (default) ranks by fused score per line; ``score`` ranks by
        raw fused score, which favours large regions.
    max_block_lines:
        Drop merged blocks longer than this - a guard against one runaway merge
        swallowing an entire file.
    agreement_bonus:
        Multiplicative boost per extra retriever that surfaced the block.
        Cross-retriever agreement is strong evidence, so a block found by all
        three is worth more than the sum of its ranks.
    drop_enclosing:
        Discard candidate spans that wholly contain other candidates (modules,
        outer classes), folding their score into the enclosed spans. Without
        this a module hit merges every result in its file into one blob.
    """
    fused = fuse_ranks(ranked_lists, k=k)
    if not fused:
        return []

    candidates: List[Tuple[str, float, Dict[str, int], CodeSpan]] = []
    for node_id, (score, prov) in fused.items():
        span = graph.spans.get(node_id)
        if span is None:
            continue
        candidates.append((node_id, score, prov, span))

    if not candidates:
        return []

    if drop_enclosing:
        candidates = _drop_enclosing(candidates)

    blocks = _merge_spans(candidates, merge_gap)

    if max_block_lines is not None:
        filtered = [b for b in blocks if b.line_count <= max_block_lines]
        # Never return nothing purely because of the size guard.
        blocks = filtered or blocks

    for block in blocks:
        if agreement_bonus and block.agreement > 1:
            block.fused_score *= 1.0 + agreement_bonus * (block.agreement - 1)
        _choose_primary(block, graph)

    if rank_by == "density":
        blocks.sort(key=lambda b: (-b.density, -b.agreement, b.path, b.start_line))
    elif rank_by == "score":
        blocks.sort(key=lambda b: (-b.fused_score, -b.agreement, b.path, b.start_line))
    else:
        raise ValueError(f"unknown rank_by: {rank_by!r} (expected 'density' or 'score')")

    return blocks[:top_n]
