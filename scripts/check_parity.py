#!/usr/bin/env python3
"""
Cross-implementation parity check.

Runs the Rust fixture emitter, then diffs its output against the Python
implementation on the same inputs. Any divergence fails the build - that is
what keeps the two ports honest as either side evolves.
"""

from __future__ import annotations

import difflib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compressors import lossless_code_compressor, lossless_terminal_cleaner  # noqa: E402
from data_converter import json_to_minimal_yaml  # noqa: E402

FIXTURES = ROOT / "fixtures"
RUST_OUT = FIXTURES / "rust_out"


def run_rust_emitter() -> bool:
    print("==> generating Rust fixture output")
    result = subprocess.run(
        ["cargo", "test", "--test", "parity_fixtures", "--", "--nocapture"],
        cwd=ROOT / "rust",
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        return False
    return True


def compare(label: str, python_text: str, rust_file: pathlib.Path) -> bool:
    if not rust_file.exists():
        print(f"  {label}: MISSING rust output ({rust_file})")
        return False

    rust_text = rust_file.read_text(encoding="utf-8")
    if python_text.strip() == rust_text.strip():
        print(f"  {label}: MATCH ({len(python_text)} chars)")
        return True

    print(f"  {label}: DIVERGED")
    diff = difflib.unified_diff(
        python_text.splitlines(),
        rust_text.splitlines(),
        fromfile=f"python/{label}",
        tofile=f"rust/{label}",
        lineterm="",
    )
    for line in list(diff)[:60]:
        print("    " + line)
    return False


def main() -> int:
    if not run_rust_emitter():
        print("cargo test failed - cannot compare")
        return 1

    print("==> comparing implementations")
    ok = True

    code = (FIXTURES / "sample_code.py").read_text(encoding="utf-8")
    ok &= compare("code", lossless_code_compressor(code), RUST_OUT / "code.txt")

    logs = (FIXTURES / "sample_logs.txt").read_text(encoding="utf-8")
    ok &= compare("logs", lossless_terminal_cleaner(logs), RUST_OUT / "logs.txt")

    data = json.loads((FIXTURES / "sample_data.json").read_text(encoding="utf-8"))
    ok &= compare("data", json_to_minimal_yaml(data), RUST_OUT / "data.txt")

    print()
    if ok:
        print("PARITY OK - both implementations produce identical output")
        return 0
    print("PARITY FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
