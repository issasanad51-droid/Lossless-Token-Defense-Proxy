"""
Local AST graph engine.

Indexes functions, classes and methods into :class:`CodeSpan` nodes and links
them into a call graph. Everything is derived from the real syntax tree, not
regex heuristics, so a "call" edge means the parser actually saw a call.

Edges
-----
``CALLS``     caller -> callee (resolved by scope, then by unique name)
``CONTAINS``  class -> method, module -> top-level def
``INHERITS``  subclass -> base class
``IMPORTS``   module -> module

Resolution is deliberately conservative: an unresolvable name produces no edge
rather than a wrong one, because a bogus edge silently corrupts PageRank.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = ["CodeSpan", "EdgeKind", "CodeGraph", "build_graph"]


class EdgeKind(str, Enum):
    CALLS = "calls"
    CONTAINS = "contains"
    INHERITS = "inherits"
    IMPORTS = "imports"


@dataclass
class CodeSpan:
    """One indexed unit of code (module, class, function or method)."""

    node_id: str
    kind: str                     # module | class | function | method
    name: str
    qualname: str
    path: str
    start_line: int               # 1-based, inclusive
    end_line: int                 # 1-based, inclusive
    source: str = ""
    docstring: str = ""
    signature: str = ""
    parent: Optional[str] = None
    decorators: List[str] = field(default_factory=list)

    @property
    def line_count(self) -> int:
        return max(1, self.end_line - self.start_line + 1)

    def overlaps(self, other: "CodeSpan") -> bool:
        """True when two spans share at least one line of the same file."""
        if self.path != other.path:
            return False
        return self.start_line <= other.end_line and other.start_line <= self.end_line

    def text_for_index(self) -> str:
        """The text a retriever should match against."""
        parts = [self.name, self.qualname.replace(".", " "), self.signature, self.docstring, self.source]
        return "\n".join(p for p in parts if p)


class CodeGraph:
    """A queryable graph of code spans and the edges between them."""

    def __init__(self) -> None:
        self.spans: Dict[str, CodeSpan] = {}
        self.edges: Dict[Tuple[str, str, EdgeKind], float] = {}
        self._out: Dict[str, List[Tuple[str, EdgeKind, float]]] = {}
        self._in: Dict[str, List[Tuple[str, EdgeKind, float]]] = {}
        self._by_name: Dict[str, List[str]] = {}

    # -- construction -------------------------------------------------------

    def add_span(self, span: CodeSpan) -> None:
        self.spans[span.node_id] = span
        self._by_name.setdefault(span.name, []).append(span.node_id)
        self._out.setdefault(span.node_id, [])
        self._in.setdefault(span.node_id, [])

    def add_edge(self, src: str, dst: str, kind: EdgeKind, weight: float = 1.0) -> None:
        if src == dst or src not in self.spans or dst not in self.spans:
            return
        key = (src, dst, kind)
        if key in self.edges:
            self.edges[key] += weight
            return
        self.edges[key] = weight
        self._out.setdefault(src, []).append((dst, kind, weight))
        self._in.setdefault(dst, []).append((src, kind, weight))

    # -- queries ------------------------------------------------------------

    def successors(self, node_id: str) -> List[Tuple[str, EdgeKind, float]]:
        return self._out.get(node_id, [])

    def predecessors(self, node_id: str) -> List[Tuple[str, EdgeKind, float]]:
        return self._in.get(node_id, [])

    def neighbors(self, node_id: str) -> Set[str]:
        return {d for d, _, _ in self.successors(node_id)} | {s for s, _, _ in self.predecessors(node_id)}

    def find_by_name(self, name: str) -> List[str]:
        """Look up by bare name or by qualified-name suffix."""
        if name in self._by_name:
            return list(self._by_name[name])
        hits = [nid for nid, s in self.spans.items() if s.qualname == name or s.qualname.endswith("." + name)]
        return hits

    def __len__(self) -> int:
        return len(self.spans)

    def stats(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for (_, _, kind) in self.edges:
            counts[kind.value] = counts.get(kind.value, 0) + 1
        return {"nodes": len(self.spans), "edges": len(self.edges), **counts}


class _ModuleIndexer(ast.NodeVisitor):
    """Walks one module, emitting spans and edges."""

    def __init__(self, graph: CodeGraph, path: str, source: str, module_id: str) -> None:
        self.graph = graph
        self.path = path
        self.lines = source.splitlines()
        self.module_id = module_id
        self.scope: List[str] = []          # qualname components
        self.node_stack: List[str] = [module_id]
        # Local symbol table: bare name -> node_id, for scope-aware resolution.
        self.local_defs: Dict[str, str] = {}
        self.pending_calls: List[Tuple[str, str]] = []   # (caller_id, callee_name)

    # -- helpers ------------------------------------------------------------

    def _segment(self, node: ast.AST) -> Tuple[int, int, str]:
        start = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", start) or start
        # Include decorators in the span so retrieved context is complete.
        for dec in getattr(node, "decorator_list", []) or []:
            start = min(start, getattr(dec, "lineno", start))
        body = "\n".join(self.lines[start - 1 : end])
        return start, end, body

    def _signature(self, node: ast.AST) -> str:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                args = ast.unparse(node.args)
            except Exception:
                args = ", ".join(a.arg for a in node.args.args)
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            return f"{prefix} {node.name}({args})"
        if isinstance(node, ast.ClassDef):
            bases = []
            for b in node.bases:
                try:
                    bases.append(ast.unparse(b))
                except Exception:
                    pass
            return f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"
        return ""

    def _decorators(self, node: ast.AST) -> List[str]:
        out = []
        for dec in getattr(node, "decorator_list", []) or []:
            try:
                out.append(ast.unparse(dec))
            except Exception:
                continue
        return out

    # -- visitors -----------------------------------------------------------

    def _visit_def(self, node: ast.AST, kind: str) -> None:
        name = getattr(node, "name", "<anon>")
        qual = ".".join(self.scope + [name])
        start, end, body = self._segment(node)
        node_id = f"{self.path}::{qual}"

        span = CodeSpan(
            node_id=node_id,
            kind=kind,
            name=name,
            qualname=qual,
            path=self.path,
            start_line=start,
            end_line=end,
            source=body,
            docstring=ast.get_docstring(node) or "" if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ) else "",
            signature=self._signature(node),
            parent=self.node_stack[-1],
            decorators=self._decorators(node),
        )
        self.graph.add_span(span)
        self.graph.add_edge(self.node_stack[-1], node_id, EdgeKind.CONTAINS)

        # Register for later call resolution, under both bare and qualified name.
        self.local_defs.setdefault(name, node_id)
        self.local_defs[qual] = node_id

        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = self._name_of(base)
                if base_name:
                    self.pending_calls.append((node_id, f"__inherits__{base_name}"))

        self.scope.append(name)
        self.node_stack.append(node_id)
        for child in ast.iter_child_nodes(node):
            self.visit(child)
        self.node_stack.pop()
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_def(node, "method" if self._inside_class() else "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_def(node, "method" if self._inside_class() else "function")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_def(node, "class")

    def _inside_class(self) -> bool:
        parent = self.node_stack[-1]
        span = self.graph.spans.get(parent)
        return bool(span and span.kind == "class")

    @staticmethod
    def _name_of(node: ast.AST) -> Optional[str]:
        """Best-effort callable name: f(), obj.method(), mod.obj.method()."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    def visit_Call(self, node: ast.Call) -> None:
        callee = self._name_of(node.func)
        if callee:
            self.pending_calls.append((self.node_stack[-1], callee))
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.pending_calls.append((self.module_id, f"__imports__{alias.name.split('.')[0]}"))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.pending_calls.append((self.module_id, f"__imports__{node.module.split('.')[0]}"))


def build_graph(
    sources: Dict[str, str],
    *,
    include_modules: bool = True,
) -> CodeGraph:
    """Build a :class:`CodeGraph` from ``{path: source}``.

    Files that fail to parse are skipped rather than aborting the index - a
    partial graph is far more useful than none.
    """
    graph = CodeGraph()
    indexers: List[_ModuleIndexer] = []

    for path, source in sorted(sources.items()):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        module_name = os.path.splitext(os.path.basename(path))[0]
        module_id = f"{path}::<module>"
        lines = source.splitlines()

        if include_modules:
            graph.add_span(
                CodeSpan(
                    node_id=module_id,
                    kind="module",
                    name=module_name,
                    qualname=module_name,
                    path=path,
                    start_line=1,
                    end_line=max(1, len(lines)),
                    source="",  # modules index by their children, not whole-file text
                    docstring=ast.get_docstring(tree) or "",
                )
            )

        indexer = _ModuleIndexer(graph, path, source, module_id)
        for child in ast.iter_child_nodes(tree):
            indexer.visit(child)
        indexers.append(indexer)

    # Resolve names to edges once every module has been indexed.
    module_by_name: Dict[str, str] = {
        s.name: nid for nid, s in graph.spans.items() if s.kind == "module"
    }

    for indexer in indexers:
        for src_id, raw_name in indexer.pending_calls:
            if raw_name.startswith("__imports__"):
                target = module_by_name.get(raw_name[len("__imports__") :])
                if target:
                    graph.add_edge(indexer.module_id, target, EdgeKind.IMPORTS)
                continue

            if raw_name.startswith("__inherits__"):
                base = raw_name[len("__inherits__") :]
                target = indexer.local_defs.get(base)
                if target is None:
                    candidates = graph.find_by_name(base)
                    target = candidates[0] if len(candidates) == 1 else None
                if target:
                    graph.add_edge(src_id, target, EdgeKind.INHERITS)
                continue

            # 1. same-module definition wins
            target = indexer.local_defs.get(raw_name)
            # 2. otherwise accept a globally unique name
            if target is None:
                candidates = graph.find_by_name(raw_name)
                target = candidates[0] if len(candidates) == 1 else None
            # 3. ambiguous -> no edge, on purpose
            if target:
                graph.add_edge(src_id, target, EdgeKind.CALLS)

    return graph
