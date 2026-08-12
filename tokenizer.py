"""
Thin tiktoken wrapper with graceful degradation.

``tiktoken`` downloads its BPE tables on first use. In locked-down CI images,
air-gapped boxes or sandboxes that blocks the CDN, that download fails - and a
token *counter* that hard-crashes is worse than one that says "I estimated".

Resolution order:
  1. the exact encoding for the requested model (e.g. gpt-4o -> o200k_base)
  2. any locally cached tiktoken encoding (cl100k_base, offline bundles, ...)
  3. a deterministic heuristic estimator (clearly flagged as approximate)

Savings percentages stay meaningful under every tier because baseline and
optimized text are always measured with the *same* counter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional

__all__ = ["TokenCounter"]

_FALLBACK_ENCODINGS = ("o200k_base", "cl100k_base", "cl100k_base_offline", "p50k_base", "gpt2")

# Rough BPE-like segmentation used only when tiktoken is unavailable.
_HEURISTIC_SPLIT = re.compile(r"""\s+|[A-Za-z]+|\d|[^\sA-Za-z\d]""")


@dataclass(frozen=True)
class _Backend:
    name: str
    exact: bool
    encode: Callable[[str], List[int]]


class TokenCounter:
    """Counts tokens for a model, degrading gracefully when offline."""

    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model
        self._backend = self._resolve(model)

    # -- resolution ---------------------------------------------------------

    @staticmethod
    def _resolve(model: str) -> _Backend:
        try:
            import tiktoken
        except ImportError:
            return _Backend("heuristic (tiktoken not installed)", False, TokenCounter._estimate)

        try:
            enc = tiktoken.encoding_for_model(model)
            return _Backend(f"tiktoken:{enc.name}", True, lambda s: enc.encode(s, disallowed_special=()))
        except Exception:
            pass

        for name in _FALLBACK_ENCODINGS:
            try:
                enc = tiktoken.get_encoding(name)
                return _Backend(
                    f"tiktoken:{name} (proxy for {model})",
                    False,
                    lambda s, e=enc: e.encode(s, disallowed_special=()),
                )
            except Exception:
                continue

        return _Backend("heuristic (no BPE data available)", False, TokenCounter._estimate)

    @staticmethod
    def _estimate(text: str) -> List[int]:
        """Deterministic stand-in: ~1 token per word-chunk, digits split out."""
        pieces = [p for p in _HEURISTIC_SPLIT.findall(text) if p.strip() or p == " "]
        tokens: List[int] = []
        for piece in pieces:
            if piece.isspace():
                tokens.extend([0] * max(1, len(piece) // 4) if len(piece) > 1 else [0])
            elif piece.isalpha() and len(piece) > 6:
                tokens.extend([0] * ((len(piece) + 3) // 4))  # long words split
            else:
                tokens.append(0)
        return tokens

    # -- public API ---------------------------------------------------------

    @property
    def backend(self) -> str:
        return self._backend.name

    @property
    def is_exact(self) -> bool:
        return self._backend.exact

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._backend.encode(text))
