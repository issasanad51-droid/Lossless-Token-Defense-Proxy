"""
Vector lookup over code spans.

This is a genuine vector space model - TF-IDF weighted sparse vectors with
cosine similarity - not a neural embedder. That is a deliberate, and honest,
choice:

* it has zero dependencies and runs offline, matching the rest of the repo
* it is deterministic, so retrieval tests assert real values instead of drifting
* code search leans heavily on exact identifier matches, where lexical scoring
  is a strong baseline

The important part is the **interface**. :class:`VectorStore` accepts an
``embedder`` callable, so swapping in OpenAI / sentence-transformers means
passing a function - no changes to the graph, PPR, RRF or fusion layers.

Tokenization is code-aware: ``parse_http_request`` also yields ``parse``,
``http`` and ``request``, and ``parseHTTPRequest`` splits on camelCase, so a
natural-language query can hit an identifier it does not literally match.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

__all__ = ["code_tokenize", "VectorStore"]

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+|_")

# Syntax noise that appears in nearly every span and discriminates nothing.
_STOPWORDS = frozenset(
    """
    the a an and or not is are was were be been being of to in for on with as by
    at from this that these those it its if else elif return def class self cls
    import none true false pass raise try except finally while do then
    """.split()
)


def _split_identifier(token: str) -> List[str]:
    """``parse_httpRequest`` -> ``[parse, http, request]``."""
    parts: List[str] = []
    for chunk in token.split("_"):
        if not chunk:
            continue
        parts.extend(p.lower() for p in _CAMEL.findall(chunk) if p and p != "_")
    return parts


def code_tokenize(text: str, *, keep_compound: bool = True) -> List[str]:
    """Tokenize code or a natural-language query into comparable terms."""
    tokens: List[str] = []
    for raw in _WORD.findall(text or ""):
        lower = raw.lower()
        pieces = _split_identifier(raw)
        if keep_compound and len(pieces) > 1:
            tokens.append(lower)          # the whole identifier
        tokens.extend(pieces)             # and its parts
        if not pieces:
            tokens.append(lower)
    return [t for t in tokens if t and t not in _STOPWORDS and len(t) > 1]


@dataclass
class _Doc:
    doc_id: str
    vector: Dict[str, float]
    norm: float


class VectorStore:
    """Sparse TF-IDF vector index with cosine similarity.

    Parameters
    ----------
    embedder:
        Optional ``Callable[[str], Sequence[float]]``. When supplied, dense
        embeddings are used instead of TF-IDF and cosine is computed over them.
    """

    def __init__(self, embedder: Optional[Callable[[str], Sequence[float]]] = None) -> None:
        self.embedder = embedder
        self._docs: Dict[str, _Doc] = {}
        self._raw_tf: Dict[str, Counter] = {}
        self._df: Counter = Counter()
        self._dense: Dict[str, Sequence[float]] = {}
        self._dirty = False

    # -- indexing -----------------------------------------------------------

    def add(self, doc_id: str, text: str) -> None:
        if self.embedder is not None:
            self._dense[doc_id] = self.embedder(text)
            return
        tf = Counter(code_tokenize(text))
        if not tf:
            tf = Counter({"": 1})
        self._raw_tf[doc_id] = tf
        for term in tf:
            self._df[term] += 1
        self._dirty = True

    def add_many(self, docs: Union[Mapping[str, str], Iterable[Tuple[str, str]]]) -> None:
        """Index a batch, given either ``{doc_id: text}`` or ``[(doc_id, text)]``."""
        pairs = docs.items() if isinstance(docs, Mapping) else docs
        for doc_id, text in pairs:
            self.add(doc_id, text)

    def _rebuild(self) -> None:
        """Recompute TF-IDF weights. Called lazily on first query."""
        n = max(1, len(self._raw_tf))
        self._docs.clear()
        for doc_id, tf in self._raw_tf.items():
            max_tf = max(tf.values()) or 1
            vec: Dict[str, float] = {}
            for term, count in tf.items():
                # Sublinear TF damps repeated identifiers; smoothed IDF avoids /0.
                tf_w = 0.5 + 0.5 * (count / max_tf)
                idf = math.log((n + 1) / (1 + self._df[term])) + 1.0
                vec[term] = tf_w * idf
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self._docs[doc_id] = _Doc(doc_id, vec, norm)
        self._dirty = False

    # -- query --------------------------------------------------------------

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Return ``[(doc_id, score)]`` sorted by descending cosine similarity."""
        if self.embedder is not None:
            return self._search_dense(query, top_k)

        if self._dirty:
            self._rebuild()
        if not self._docs:
            return []

        q_tf = Counter(code_tokenize(query))
        if not q_tf:
            return []

        n = max(1, len(self._docs))
        max_tf = max(q_tf.values()) or 1
        q_vec: Dict[str, float] = {}
        for term, count in q_tf.items():
            if term not in self._df:
                continue
            tf_w = 0.5 + 0.5 * (count / max_tf)
            idf = math.log((n + 1) / (1 + self._df[term])) + 1.0
            q_vec[term] = tf_w * idf
        if not q_vec:
            return []
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

        scored: List[Tuple[str, float]] = []
        for doc in self._docs.values():
            # Iterate the shorter side; query vectors are tiny next to documents.
            dot = sum(w * doc.vector.get(term, 0.0) for term, w in q_vec.items())
            if dot > 0:
                scored.append((doc.doc_id, dot / (q_norm * doc.norm)))

        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored[:top_k]

    def _search_dense(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        assert self.embedder is not None
        q = self.embedder(query)
        q_norm = math.sqrt(sum(x * x for x in q)) or 1.0
        scored: List[Tuple[str, float]] = []
        for doc_id, vec in self._dense.items():
            dot = sum(a * b for a, b in zip(q, vec))
            d_norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            scored.append((doc_id, dot / (q_norm * d_norm)))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._dense) if self.embedder is not None else len(self._raw_tf)
