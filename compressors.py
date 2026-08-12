"""
Lossless compressors for source code and terminal output.

"Lossless" here has a precise, testable meaning:

* ``lossless_code_compressor``  -> the compressed source parses to an
  **identical AST** as the input. Only human-facing comments and redundant
  blank space are removed. Indentation, string contents (including
  docstrings) and every executable token are preserved byte-for-byte.
* ``lossless_terminal_cleaner`` -> every line carrying *signal* (errors,
  failures, completion markers) survives. Only mechanical progress noise
  (tickers, percentages, download loops, spinner frames) is dropped, and the
  drop is accounted for so nothing silently disappears.

Both functions are pure: same input -> same output, no globals, no I/O.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "lossless_code_compressor",
    "lossless_terminal_cleaner",
]


# ---------------------------------------------------------------------------
# Code compression
# ---------------------------------------------------------------------------

# Comments that are NOT human prose: they change how tools/interpreters behave.
# Stripping these would be lossy, so they are always kept.
_SEMANTIC_COMMENT = re.compile(
    r"""^\#\s*(?:
          !                      # shebang
        | -\*-                   # -*- coding: ... -*-
        | (?:type|mypy|pyright)\b
        | noqa\b
        | pragma\b
        | pylint\b
        | flake8\b
        | ruff\b
        | fmt\s*:\s*(?:on|off|skip)
        | yapf\s*:\s*(?:disable|enable)
        | isort\s*:\s*
        | nosec\b
        | coding[:=]
    )""",
    re.VERBOSE | re.IGNORECASE,
)


def _scan_line(line: str, in_triple: Optional[str]) -> Tuple[Optional[int], Optional[str]]:
    """Scan one physical line, tracking string state.

    Returns ``(comment_start, new_triple_state)`` where ``comment_start`` is the
    index of the ``#`` that begins a real comment, or ``None`` if this line has
    no comment. A ``#`` inside a string literal is never reported.
    """
    i = 0
    n = len(line)

    # Continue an open triple-quoted string from a previous line.
    if in_triple is not None:
        while i < n:
            if line[i] == "\\":
                i += 2
                continue
            if line.startswith(in_triple, i):
                i += len(in_triple)
                in_triple = None
                break
            i += 1
        else:
            return None, in_triple  # still inside the string

    while i < n:
        ch = line[i]

        if ch == "#":
            return i, None

        if ch in ("'", '"'):
            triple = line[i : i + 3]
            if triple in ("'''", '"""'):
                i += 3
                # Does it close on this same line?
                while i < n:
                    if line[i] == "\\":
                        i += 2
                        continue
                    if line.startswith(triple, i):
                        i += len(triple)
                        break
                    i += 1
                else:
                    return None, triple  # spills onto the next line
                continue

            # Single-quoted string.
            quote = ch
            i += 1
            while i < n:
                if line[i] == "\\":
                    i += 2
                    continue
                if line[i] == quote:
                    i += 1
                    break
                i += 1
            continue

        i += 1

    return None, None


def lossless_code_compressor(
    raw_code: str,
    *,
    max_consecutive_blank_lines: int = 1,
    strip_trailing_comments: bool = True,
    keep_semantic_comments: bool = True,
) -> str:
    """Strip human comments and collapse blank-line runs, losslessly.

    Parameters
    ----------
    raw_code:
        Source text to compress.
    max_consecutive_blank_lines:
        Runs of blank lines longer than this are collapsed down to it.
        ``0`` removes blank lines between statements entirely.
    strip_trailing_comments:
        Also remove ``code  # comment`` tails, not just whole-line comments.
    keep_semantic_comments:
        Preserve shebangs, coding cookies and tool directives (``# noqa``,
        ``# type:``, ``# fmt: off`` ...), which are not human prose.

    Guarantees
    ----------
    * Never edits anything inside a string literal (docstrings included).
    * Never changes indentation of a kept line.
    * Blank lines inside triple-quoted strings are left untouched.
    """
    if not raw_code:
        return ""

    lines = raw_code.splitlines()
    out: List[str] = []
    in_triple: Optional[str] = None
    blank_run = 0

    for lineno, line in enumerate(lines, start=1):
        was_in_string = in_triple is not None
        comment_start, in_triple = _scan_line(line, in_triple)

        # Lines that live inside a multi-line string are sacred: emit verbatim.
        if was_in_string:
            out.append(line)
            blank_run = 0
            continue

        if comment_start is not None:
            comment_text = line[comment_start:]
            stripped_before = line[:comment_start].strip()
            is_whole_line_comment = stripped_before == ""

            keep_it = keep_semantic_comments and (
                _SEMANTIC_COMMENT.match(comment_text.strip())
                or (lineno == 1 and comment_text.startswith("#!"))
            )

            if keep_it:
                out.append(line.rstrip())
                blank_run = 0
                continue

            if is_whole_line_comment:
                continue  # drop the line entirely

            if strip_trailing_comments:
                line = line[:comment_start].rstrip()
            else:
                line = line.rstrip()
        else:
            line = line.rstrip()

        if line.strip() == "":
            blank_run += 1
            if blank_run > max_consecutive_blank_lines:
                continue
            out.append("")
            continue

        blank_run = 0
        out.append(line)

    # Trim leading/trailing blank lines produced by removals.
    while out and out[0].strip() == "":
        out.pop(0)
    while out and out[-1].strip() == "":
        out.pop()

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Terminal / build-log cleaning
# ---------------------------------------------------------------------------

# Signal always wins: if a line matches these, it is kept no matter what.
_SIGNAL_PATTERNS: Sequence[re.Pattern] = (
    re.compile(r"\b(?:error|errors)\b", re.IGNORECASE),
    re.compile(r"\b(?:fatal|panic|abort(?:ed)?|segfault|core dumped)\b", re.IGNORECASE),
    re.compile(r"\b(?:fail|failed|failure|failing)\b", re.IGNORECASE),
    re.compile(r"\b(?:exception|traceback|stack ?trace|assertion)\b", re.IGNORECASE),
    re.compile(r"\b(?:undefined reference|cannot find|not found|no such file)\b", re.IGNORECASE),
    re.compile(r"\b(?:exit(?:ed)? (?:code|status)|exit code)\b", re.IGNORECASE),
    re.compile(r"\b(?:succeeded|success|successful|completed successfully)\b", re.IGNORECASE),
    re.compile(r"\b(?:build (?:finished|complete|succeeded|failed))\b", re.IGNORECASE),
    re.compile(r"\b(?:finished|done) in\b", re.IGNORECASE),
    re.compile(r"^\s*(?:E|ERROR|FAIL|FAILED|CRITICAL)\b"),
    re.compile(r"\bwarning\b.*\b(?:treated as error|-Werror)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:at |File \"|\s+\^+\s*$)"),  # stack frames / carets
)

# Pure mechanical noise: progress, downloads, spinners.
_NOISE_PATTERNS: Sequence[re.Pattern] = (
    re.compile(r"^\s*\[\s*\d+\s*/\s*\d+\s*\]"),                  # [1/250] Compiling foo.c
    re.compile(r"^\s*\(\s*\d+\s*/\s*\d+\s*\)"),                  # (17/250) Linking
    re.compile(r"^\s*\d{1,3}\s*%"),                              # 45% completed
    re.compile(r"\b\d{1,3}(?:\.\d+)?\s*%\s*(?:complete|completed|done|finished)?\b", re.IGNORECASE),
    re.compile(r"^\s*(?:Compiling|Building|Downloading|Fetching|Extracting|Unpacking|"
               r"Installing|Resolving|Linking|Indexing|Uploading|Pulling|Cloning)\b", re.IGNORECASE),
    re.compile(r"\b(?:ETA|eta)\s+\d"),                           # ETA 00:12
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:[KMG]i?B)\s*/\s*\d+(?:\.\d+)?\s*(?:[KMG]i?B)"),  # 4.2MB/10MB
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:[KMG]i?B|B)/s\b"),         # 3.4MB/s
    re.compile(r"^[\s\|\-\\/\*\.=#>â–ˆâ–‘â–’â–“â–ºâ ‹â ™â ¹â ¸â ¼â ´â ¦â §â ‡â ]+$"),           # spinner / bar frames
    re.compile(r"^\s*(?:\[=*>?\s*\]|\[#*\s*\]|\[\.*\s*\])\s*$"),  # [====>   ]
    re.compile(r"^\s*(?:Progress|Status):", re.IGNORECASE),
    re.compile(r"^\s*(?:Receiving|Counting|Compressing|Delta) objects:", re.IGNORECASE),
    re.compile(r"^\s*remote:\s*(?:Counting|Compressing|Enumerating|Total)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:Reading (?:package|state)|Get:\d+|Hit:\d+|Ign:\d+)\b"),
    re.compile(r"^\s*(?:npm|yarn|pnpm)\s+(?:WARN\s+)?(?:idealTree|timing|sill|http fetch)\b"),
    re.compile(r"^\s*\.{3,}\s*$"),
)

_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07")
_CR_FRAMES = re.compile(r"^.*\r(?!\n)")  # keep only the final frame of a \r-redrawn line


def _is_signal(line: str) -> bool:
    return any(p.search(line) for p in _SIGNAL_PATTERNS)


def _is_noise(line: str) -> bool:
    return any(p.search(line) for p in _NOISE_PATTERNS)


def lossless_terminal_cleaner(
    raw_logs: str,
    *,
    annotate_drops: bool = True,
    collapse_duplicates: bool = True,
) -> str:
    """Strip build tickers and download loops, keep failures and completions.

    Signal beats noise: a line matching an error/completion pattern is kept even
    if it also looks like a progress ticker (e.g. ``[42/250] error: ...``).

    Parameters
    ----------
    annotate_drops:
        Insert a compact ``... <N> progress lines omitted ...`` marker for each
        removed run, so the model still knows work happened (and how much)
        without paying for 250 near-identical lines.
    collapse_duplicates:
        Fold runs of identical consecutive kept lines into ``line  (xN)``.
    """
    if not raw_logs:
        return ""

    out: List[str] = []
    dropped_run = 0
    prev_line: Optional[str] = None
    prev_count = 0

    def flush_duplicates() -> None:
        nonlocal prev_line, prev_count
        if prev_line is None:
            return
        if prev_count > 1:
            out.append(f"{prev_line}  (x{prev_count})")
        else:
            out.append(prev_line)
        prev_line, prev_count = None, 0

    def flush_dropped() -> None:
        nonlocal dropped_run
        if dropped_run and annotate_drops:
            out.append(f"... {dropped_run} progress lines omitted ...")
        dropped_run = 0

    for raw_line in raw_logs.splitlines():
        # Collapse carriage-return redraws down to the last rendered frame,
        # then remove ANSI colour/cursor escapes.
        line = _CR_FRAMES.sub("", raw_line)
        line = _ANSI.sub("", line).rstrip()

        if line.strip() == "":
            continue

        if _is_signal(line) or not _is_noise(line):
            flush_dropped()
            if collapse_duplicates and line == prev_line:
                prev_count += 1
            else:
                flush_duplicates()
                prev_line, prev_count = line, 1
            continue

        # Noise.
        flush_duplicates()
        dropped_run += 1

    flush_duplicates()
    flush_dropped()

    return "\n".join(out)
