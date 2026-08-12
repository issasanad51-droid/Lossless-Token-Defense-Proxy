"""
Losslessness proofs.

These are the tests that matter. Anyone can delete lines and call it
"compression" - the point here is that meaning survives:

* code   -> compressed source must parse to an IDENTICAL AST
* data   -> flattened form must round-trip back to the EXACT original object
* logs   -> every error/failure/completion line must still be present

Run:  pytest -q
"""

from __future__ import annotations

import ast
import json

import pytest

from compressors import lossless_code_compressor, lossless_terminal_cleaner
from data_converter import json_to_minimal_yaml, minimal_yaml_to_json
from orchestrator import DUMMY_CODE, DUMMY_LOGS, DUMMY_SYSTEM_DATA, TokenDefenseProxy


# ---------------------------------------------------------------------------
# Code compressor
# ---------------------------------------------------------------------------

def assert_ast_identical(original: str, compressed: str) -> None:
    assert ast.dump(ast.parse(original)) == ast.dump(ast.parse(compressed))


def test_demo_code_ast_is_preserved():
    assert_ast_identical(DUMMY_CODE, lossless_code_compressor(DUMMY_CODE))


def test_comments_are_removed():
    out = lossless_code_compressor(DUMMY_CODE)
    assert "TODO: refactor this someday" not in out
    assert "Store the gateway client" not in out
    assert "the ISO currency code" not in out


def test_hash_inside_string_survives():
    src = 'note = "refund # not a comment"  # real comment\n'
    out = lossless_code_compressor(src)
    assert out == 'note = "refund # not a comment"'
    assert_ast_identical(src, out)


def test_docstring_is_untouched():
    src = '''def f():
    """Line one.

    Indented   body  with   spacing.


    """
    return 1
'''
    out = lossless_code_compressor(src)
    assert "Indented   body  with   spacing." in out
    assert_ast_identical(src, out)
    assert ast.get_docstring(ast.parse(out).body[0]) == ast.get_docstring(ast.parse(src).body[0])


def test_hash_in_single_quoted_and_fstring():
    src = "a = '# not a comment'\nb = f'{x}#tag'  # gone\n"
    out = lossless_code_compressor(src)
    assert "'# not a comment'" in out
    assert "#tag" in out
    assert "# gone" not in out
    assert_ast_identical(src, out)


def test_semantic_comments_are_kept():
    src = (
        "#!/usr/bin/env python3\n"
        "# -*- coding: utf-8 -*-\n"
        "# just chatter\n"
        "import os  # noqa: F401\n"
        "x = 1  # type: int\n"
    )
    out = lossless_code_compressor(src)
    assert "#!/usr/bin/env python3" in out
    assert "coding: utf-8" in out
    assert "just chatter" not in out
    assert "noqa" in out
    assert "type: int" in out


def test_blank_line_runs_collapse():
    src = "a = 1\n\n\n\n\n\nb = 2\n"
    assert lossless_code_compressor(src) == "a = 1\n\nb = 2"
    assert lossless_code_compressor(src, max_consecutive_blank_lines=0) == "a = 1\nb = 2"


def test_indentation_is_preserved_exactly():
    src = "if x:\n    if y:\n        deep()\n"
    out = lossless_code_compressor(src)
    assert "        deep()" in out
    assert_ast_identical(src, out)


def test_blank_lines_inside_triple_string_survive():
    src = 'x = """a\n\n\n\n\nb"""\n'
    out = lossless_code_compressor(src)
    assert ast.parse(out).body[0].value.value == "a\n\n\n\n\nb"


def test_empty_input():
    assert lossless_code_compressor("") == ""
    assert lossless_terminal_cleaner("") == ""


@pytest.mark.parametrize("src", [
    "x = 1\n",
    "def f(a, b=2, *args, **kw):\n    return a\n",
    "class A:\n    pass\n",
    "async def g():\n    await h()\n",
    "y = [i for i in range(10) if i % 2]\n",
    "with open('f') as fh:\n    data = fh.read()\n",
    "try:\n    x()\nexcept (A, B) as e:\n    raise\nfinally:\n    z()\n",
    "s = '''triple # hash'''\n",
    "m = {'k': 'v#'}  # trailing\n",
    "lambda_fn = lambda x: x + 1\n",
    "@decorator\ndef d():\n    ...\n",
])
def test_ast_preserved_across_constructs(src):
    assert_ast_identical(src, lossless_code_compressor(src))


def test_compressor_is_idempotent():
    once = lossless_code_compressor(DUMMY_CODE)
    assert lossless_code_compressor(once) == once


# ---------------------------------------------------------------------------
# Terminal cleaner
# ---------------------------------------------------------------------------

def test_progress_noise_is_stripped():
    out = lossless_terminal_cleaner(DUMMY_LOGS)
    assert "[  1/250] Compiling src/alloc.c" not in out
    assert "12% completed" not in out
    assert "3.4MB/s" not in out


def test_errors_always_survive():
    out = lossless_terminal_cleaner(DUMMY_LOGS)
    for must_keep in [
        "has no member named 'free_list'",
        "undefined reference to symbol 'mem_pool_init'",
        "ld returned 1 exit status",
        "Error 2",
        "Build failed in 48.2s",
    ]:
        assert must_keep in out, f"lost signal line: {must_keep}"


def test_signal_beats_noise_on_same_line():
    logs = "[42/250] error: everything is on fire\n[43/250] Compiling ok.c\n"
    out = lossless_terminal_cleaner(logs)
    assert "everything is on fire" in out
    assert "Compiling ok.c" not in out


def test_drops_are_annotated():
    out = lossless_terminal_cleaner(DUMMY_LOGS)
    assert "progress lines omitted" in out


def test_carriage_return_frames_collapse():
    logs = "downloading 10%\rdownloading 50%\rdownloading 100%\nDone in 3s\n"
    out = lossless_terminal_cleaner(logs)
    assert "10%" not in out
    assert "Done in 3s" in out


def test_ansi_escapes_removed():
    out = lossless_terminal_cleaner("\x1b[31mfatal: broken\x1b[0m\n")
    assert out == "fatal: broken"


def test_duplicate_lines_collapse():
    out = lossless_terminal_cleaner("same error\nsame error\nsame error\n")
    assert out == "same error  (x3)"


def test_cleaner_actually_shrinks():
    assert len(lossless_terminal_cleaner(DUMMY_LOGS)) < len(DUMMY_LOGS) / 2


# ---------------------------------------------------------------------------
# Data converter
# ---------------------------------------------------------------------------

def test_demo_data_round_trips_exactly():
    flat = json_to_minimal_yaml(DUMMY_SYSTEM_DATA)
    assert minimal_yaml_to_json(flat) == DUMMY_SYSTEM_DATA


def test_output_has_no_json_syntax_noise():
    flat = json_to_minimal_yaml(DUMMY_SYSTEM_DATA)
    assert "{" not in flat.replace("{}", "")
    assert "[" not in flat.replace("[]", "")
    assert "payment-api" in flat
    # Unambiguous strings lose their quotes entirely...
    assert "service.name payment-api" in flat
    assert "dependencies.0.status healthy" in flat
    # ...while numeric-looking strings KEEP them, because dropping the quotes
    # would turn the string "15.4" into the float 15.4 on the way back.
    assert 'dependencies.0.version "15.4"' in flat
    assert minimal_yaml_to_json(flat)["dependencies"][0]["version"] == "15.4"


def test_quoting_is_minimal():
    """Quotes appear only where ambiguity would otherwise be introduced."""
    flat = json_to_minimal_yaml({"plain": "hello", "numeric_str": "42", "real_num": 42})
    assert "plain hello" in flat
    assert 'numeric_str "42"' in flat
    assert "real_num 42" in flat


@pytest.mark.parametrize("obj", [
    {"a": 1},
    {"a": {"b": {"c": {"d": "deep"}}}},
    {"list": [1, 2, 3]},
    {"objs": [{"x": 1}, {"x": 2}]},
    {"empty_dict": {}, "empty_list": []},
    {"nulls": None, "t": True, "f": False},
    {"num_str": "123", "bool_str": "true", "null_str": "null"},
    {"spaced": "  padded  ", "empty": ""},
    {"unicode": "caf\u00e9 \u2192 \u2603"},
    {"floats": [1.5, -0.25, 1e10]},
    {"key.with.dots": "value"},
    {"nested": {"mixed": [1, "two", True, None]}},
])
def test_round_trip_property(obj):
    assert minimal_yaml_to_json(json_to_minimal_yaml(obj)) == obj


def test_accepts_json_string():
    assert json_to_minimal_yaml('{"a": 1}') == "a 1"


def test_beats_pretty_json_on_size():
    flat = json_to_minimal_yaml(DUMMY_SYSTEM_DATA)
    pretty = json.dumps(DUMMY_SYSTEM_DATA, indent=2)
    assert len(flat) < len(pretty)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def test_pipeline_saves_tokens():
    proxy = TokenDefenseProxy()
    r = proxy.optimize_payload(DUMMY_CODE, DUMMY_LOGS, DUMMY_SYSTEM_DATA, verbose=False)
    assert r.initial_tokens > 0
    assert r.tokens_saved > 0
    assert 0 < r.percent_saved < 100
    assert r.compression_ratio > 1


def test_guardrail_is_injected():
    proxy = TokenDefenseProxy()
    r = proxy.optimize_payload(DUMMY_CODE, verbose=False)
    assert "OUTPUT PROTOCOL" in r.payload
    assert r.guardrail_tokens > 0


def test_every_section_shrinks():
    proxy = TokenDefenseProxy()
    r = proxy.optimize_payload(DUMMY_CODE, DUMMY_LOGS, DUMMY_SYSTEM_DATA, verbose=False)
    for name, s in r.section_stats.items():
        assert s["saved"] > 0, f"{name} did not shrink"


def test_handles_empty_input_gracefully():
    r = TokenDefenseProxy().optimize_payload(verbose=False)
    assert r.initial_tokens == 0
    assert r.percent_saved == 0.0


def test_report_serializes():
    r = TokenDefenseProxy().optimize_payload(DUMMY_CODE, verbose=False)
    json.dumps(r.to_dict())  # must not raise


def test_payload_retains_the_error_the_model_needs():
    r = TokenDefenseProxy().optimize_payload(DUMMY_CODE, DUMMY_LOGS, DUMMY_SYSTEM_DATA, verbose=False)
    assert "mem_pool_init" in r.payload
    assert "free_list" in r.payload
