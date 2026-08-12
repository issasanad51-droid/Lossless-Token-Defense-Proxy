"""
Personalized PageRank over the code graph.

Given one or more "active" query nodes, PPR answers: *which code is structurally
important relative to what the user is looking at?* A random surfer walks the
call graph but teleports back to the seed set with probability ``1 - damping``,
so mass concentrates around the seeds' dependency neighbourhood instead of the
globally popular utility functions.

Implemented as power iteration on adjacency lists - O(edges) per iteration, no
numpy, converges in ~20-40 iterations at damping 0.85.

Two details that matter for correctness:

* **Dangling nodes** (leaves with no outbound edges) would otherwise leak
  probability mass and silently deflate every score; their mass is redistributed
  to the seeds.
* **Edge kinds carry different weight.** A call is a stronger signal of runtime
  dependency than containment, so edges are weighted by type before
  normalization, and traversal is bidirectional (callers are as relevant as
  callees when you are debugging).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from retrieval.ast_graph import CodeGraph, EdgeKind

__all__ = ["personalized_pagerank", "ppr_rank", "DEFAULT_EDGE_WEIGHTS"]


# Relative importance of each relation when propagating structural weight.
DEFAULT_EDGE_WEIGHTS: Mapping[EdgeKind, float] = {
    EdgeKind.CALLS: 1.0,
    EdgeKind.INHERITS: 0.8,
    EdgeKind.CONTAINS: 0.5,
    EdgeKind.IMPORTS: 0.3,
}

# A backward edge is real signal but weaker than the forward dependency.
REVERSE_DAMPENING = 0.6


def _build_transition(
    graph: CodeGraph,
    edge_weights: Mapping[EdgeKind, float],
    bidirectional: bool,
) -> Dict[str, List[Tuple[str, float]]]:
    """Row-normalized transition lists: ``node -> [(neighbor, probability)]``."""
    raw: Dict[str, Dict[str, float]] = {nid: {} for nid in graph.spans}

    for (src, dst, kind), weight in graph.edges.items():
        w = edge_weights.get(kind, 0.5) * weight
        if w <= 0:
            continue
        raw[src][dst] = raw[src].get(dst, 0.0) + w
        if bidirectional:
            back = w * REVERSE_DAMPENING
            raw[dst][src] = raw[dst].get(src, 0.0) + back

    transition: Dict[str, List[Tuple[str, float]]] = {}
    for node, targets in raw.items():
        total = sum(targets.values())
        if total <= 0:
            transition[node] = []
            continue
        transition[node] = [(t, w / total) for t, w in targets.items()]
    return transition


def personalized_pagerank(
    graph: CodeGraph,
    seeds: Mapping[str, float] | Sequence[str],
    *,
    damping: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
    edge_weights: Optional[Mapping[EdgeKind, float]] = None,
    bidirectional: bool = True,
) -> Dict[str, float]:
    """Compute PPR scores for every node, personalized on ``seeds``.

    Parameters
    ----------
    seeds:
        Either a sequence of node ids (uniform weight) or a mapping of
        ``node_id -> weight``. Weights are normalized to sum to 1.
    damping:
        Probability of following an edge; ``1 - damping`` is the teleport-to-seed
        probability.

    Returns
    -------
    ``{node_id: score}`` summing to ~1.0. An empty/unknown seed set returns an
    empty dict rather than silently degrading to global PageRank - if the caller
    asked for personalization, giving them something else would be a lie.
    """
    if not graph.spans:
        return {}

    if isinstance(seeds, Mapping):
        seed_weights = {k: float(v) for k, v in seeds.items() if k in graph.spans and v > 0}
    else:
        seed_weights = {s: 1.0 for s in seeds if s in graph.spans}

    if not seed_weights:
        return {}

    total_seed = sum(seed_weights.values())
    personalization = {k: v / total_seed for k, v in seed_weights.items()}

    edge_weights = edge_weights or DEFAULT_EDGE_WEIGHTS
    transition = _build_transition(graph, edge_weights, bidirectional)

    nodes = list(graph.spans)
    scores: Dict[str, float] = {n: personalization.get(n, 0.0) for n in nodes}

    for _ in range(max_iterations):
        nxt: Dict[str, float] = {n: 0.0 for n in nodes}
        dangling_mass = 0.0

        for node, score in scores.items():
            if score == 0.0:
                continue
            out = transition.get(node)
            if not out:
                dangling_mass += score      # leaf: hold the mass, redistribute below
                continue
            for target, prob in out:
                nxt[target] += score * prob

        delta = 0.0
        for node in nodes:
            teleport = personalization.get(node, 0.0)
            # Dangling mass returns to the personalization distribution, which
            # keeps the vector stochastic and the ranking personalized.
            value = damping * (nxt[node] + dangling_mass * teleport) + (1.0 - damping) * teleport
            delta += abs(value - scores[node])
            nxt[node] = value

        scores = nxt
        if delta < tolerance:
            break

    return scores


def ppr_rank(
    graph: CodeGraph,
    seeds: Mapping[str, float] | Sequence[str],
    *,
    top_k: int = 20,
    exclude_seeds: bool = False,
    **kwargs,
) -> List[Tuple[str, float]]:
    """PPR as a ranked list, ready for fusion."""
    scores = personalized_pagerank(graph, seeds, **kwargs)
    if not scores:
        return []

    seed_ids = set(seeds.keys() if isinstance(seeds, Mapping) else seeds)
    items = [
        (nid, sc)
        for nid, sc in scores.items()
        if sc > 0 and not (exclude_seeds and nid in seed_ids)
    ]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return items[:top_k]
