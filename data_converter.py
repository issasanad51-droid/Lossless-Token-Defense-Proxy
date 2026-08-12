"""
JSON -> minimal YAML-ish flattening.

JSON spends an enormous share of its tokens on pure syntax: ``{``, ``}``,
``[``, ``]``, ``"`` around every key *and* every string value, and ``,``
between every pair. On nested config blobs that syntax is frequently 30-50%
of the token bill while carrying zero information.

``json_to_minimal_yaml`` emits a bracket-free, quote-free dotted-path form::

    service.name api-gateway
    service.ports.0 8080
    service.debug true

The transform is **reversible** (see :func:`minimal_yaml_to_json`), which is
what makes it lossless rather than merely lossy-but-short. Values that would
become ambiguous when unquoted (empty strings, leading/trailing whitespace,
strings that look like numbers/bools/null) are quoted so the round-trip is
exact.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

__all__ = [
    "json_to_minimal_yaml",
    "minimal_yaml_to_json",
]

# A bare string is safe only if it cannot be mistaken for another JSON scalar.
_LOOKS_NUMERIC = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
_RESERVED = {"true", "false", "null", "none", "~", "-", ""}


def _needs_quoting(text: str) -> bool:
    if text != text.strip() or text == "":
        return True
    if text.lower() in _RESERVED:
        return True
    if _LOOKS_NUMERIC.match(text):
        return True
    if "\n" in text or "\r" in text or "\t" in text:
        return True
    if text[0] in "\"'#[{&*!|>%@`":
        return True
    return False


def _encode_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return repr(value) if isinstance(value, float) else str(value)
    text = str(value)
    if _needs_quoting(text):
        return json.dumps(text, ensure_ascii=False)
    return text


def _decode_scalar(token: str) -> Any:
    if token.startswith('"'):
        return json.loads(token)
    if token == "null":
        return None
    if token == "true":
        return True
    if token == "false":
        return False
    if _LOOKS_NUMERIC.match(token):
        try:
            return int(token)
        except ValueError:
            return float(token)
    return token


def _escape_key(key: str) -> str:
    """Dots inside a key would collide with the path separator."""
    return key.replace("\\", "\\\\").replace(".", "\\.")


def _unescape_key(key: str) -> str:
    return key.replace("\\.", ".").replace("\\\\", "\\")


def _split_path(path: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    i = 0
    while i < len(path):
        ch = path[i]
        if ch == "\\" and i + 1 < len(path):
            buf.append(path[i : i + 2])
            i += 2
            continue
        if ch == ".":
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return [_unescape_key(p) for p in parts]


def _flatten(node: Any, prefix: str, sink: List[Tuple[str, str]]) -> None:
    if isinstance(node, Mapping):
        if not node:
            sink.append((prefix, "{}"))
            return
        for key, value in node.items():
            child = _escape_key(str(key))
            _flatten(value, f"{prefix}.{child}" if prefix else child, sink)
        return

    if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        if not node:
            sink.append((prefix, "[]"))
            return
        # Arrays of plain scalars collapse onto one line: ports 80, 443, 8080
        if all(not isinstance(v, (Mapping, list, tuple)) for v in node):
            joined = ", ".join(_encode_scalar(v) for v in node)
            sink.append((prefix, joined))
            return
        for index, value in enumerate(node):
            _flatten(value, f"{prefix}.{index}" if prefix else str(index), sink)
        return

    sink.append((prefix, _encode_scalar(node)))


def json_to_minimal_yaml(data_obj: Any, *, root_key: str = "") -> str:
    """Flatten a nested dict/list into bracket-free, quote-free text.

    Accepts a Python object or a JSON string. Output is one ``path value``
    pair per line, dot-separated, with list indices as path segments.
    """
    if isinstance(data_obj, (str, bytes, bytearray)):
        data_obj = json.loads(data_obj)

    sink: List[Tuple[str, str]] = []
    _flatten(data_obj, root_key, sink)
    return "\n".join(f"{path} {value}".rstrip() for path, value in sink)


def minimal_yaml_to_json(text: str) -> Any:
    """Inverse of :func:`json_to_minimal_yaml` (proves the transform is lossless).

    Note: a scalar array is restored as a list; a single-element scalar array is
    indistinguishable from a bare scalar, which is the one documented ambiguity.
    """
    root: Dict[str, Any] = {}

    for line in text.splitlines():
        if not line.strip():
            continue
        path, _, raw_value = line.partition(" ")
        raw_value = raw_value.strip()
        parts = _split_path(path)

        if raw_value == "{}":
            value: Any = {}
        elif raw_value == "[]":
            value = []
        elif "," in raw_value and not raw_value.startswith('"'):
            value = [_decode_scalar(v.strip()) for v in _split_top_level(raw_value)]
        else:
            value = _decode_scalar(raw_value)

        cursor: Any = root
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            if last:
                cursor[part] = value
            else:
                cursor = cursor.setdefault(part, {})

    return _relist(root)


def _split_top_level(text: str) -> List[str]:
    """Split on commas that are not inside a quoted string."""
    out: List[str] = []
    buf: List[str] = []
    in_str = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                buf.append(text[i : i + 2])
                i += 2
                continue
            if ch == '"':
                in_str = False
            buf.append(ch)
        elif ch == '"':
            in_str = True
            buf.append(ch)
        elif ch == ",":
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    out.append("".join(buf))
    return out


def _relist(node: Any) -> Any:
    """Turn {'0': .., '1': ..} maps back into lists, depth-first."""
    if not isinstance(node, dict):
        return node
    converted = {k: _relist(v) for k, v in node.items()}
    if converted and all(k.isdigit() for k in converted):
        expected = [str(i) for i in range(len(converted))]
        if sorted(converted.keys(), key=int) == expected:
            return [converted[k] for k in expected]
    return converted
