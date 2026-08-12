"""Runnable demo: python -m retrieval [query ...]

Indexes this repository's own Python files and answers queries with the top 3
densest code blocks, showing which retrievers agreed on each one.
"""

from __future__ import annotations

import pathlib
import sys

from retrieval.hybrid import HybridRetriever

DEFAULT_QUERIES = [
    "count tokens with tiktoken",
    "flatten nested json into minimal yaml",
    "remove build progress tickers from logs",
]


def load_repo_sources(root: pathlib.Path) -> dict:
    sources = {}
    skip = {".venv", ".git", "target", "__pycache__", "fixtures"}
    for path in sorted(root.rglob("*.py")):
        if any(part in skip for part in path.parts):
            continue
        try:
            sources[str(path.relative_to(root))] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return sources


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    sources = load_repo_sources(root)
    if not sources:
        print("no python sources found", file=sys.stderr)
        return 1

    print("=" * 78)
    print("HYBRID STRUCTURAL RETRIEVAL - vector + AST + PPR, fused with RRF")
    print("=" * 78)

    retriever = HybridRetriever(sources)
    stats = retriever.stats()
    total_lines = sum(len(s.splitlines()) for s in sources.values())
    print(
        f"indexed {len(sources)} files / {total_lines} lines -> "
        f"{stats['nodes']} nodes, {stats['edges']} edges "
        f"(contains={stats['contains']} calls={stats['calls']} imports={stats['imports']})"
    )

    queries = sys.argv[1:] or DEFAULT_QUERIES
    for query in queries:
        result = retriever.retrieve(query)
        print("\n" + "-" * 78)
        print(result.explain())
        if result.blocks:
            kept = result.total_lines
            pct = 100.0 * (1 - kept / total_lines) if total_lines else 0.0
            print(f"  context: {kept} lines of {total_lines} ({pct:.1f}% of the repo skipped)")

    print("\n" + "=" * 78)
    print("Top block for the last query:")
    print("=" * 78)
    if queries:
        last = retriever.retrieve(queries[-1])
        if last.blocks:
            print(last.to_context(sources).split("\n\n")[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
