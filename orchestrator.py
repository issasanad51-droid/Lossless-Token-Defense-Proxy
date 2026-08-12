#!/usr/bin/env python3
"""
TokenDefenseProxy - the middleware pipeline.

Takes raw code + raw build logs + a system-state object, runs each through the
matching lossless compressor, bolts on a hard behavioural guardrail, and
reports exactly how many tokens the whole operation saved.

Run it:

    python orchestrator.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from compressors import lossless_code_compressor, lossless_terminal_cleaner
from data_converter import json_to_minimal_yaml
from tokenizer import TokenCounter

__all__ = ["TokenDefenseProxy", "OptimizationReport"]


# ---------------------------------------------------------------------------
# The guardrail ("Ponytail / Caveman" protocol)
# ---------------------------------------------------------------------------
#
# Two jobs, and the *output* side is where the real money is: a chatty model
# burns far more tokens explaining a patch than the patch itself costs.
#   - Ponytail: tie it all back. Surgical diffs only.
#   - Caveman:  grunt, don't chat. No pleasantries, no preamble, no epilogue.

CAVEMAN_GUARDRAIL = """\
OUTPUT PROTOCOL (STRICT - violations make the response unusable):
1 Reply ONLY with unified diffs or exact replacement blocks. No prose around them.
2 BANNED openers: greetings, "Certainly", "Great question", "I'd be happy to",
  restating the task, summarizing what you are about to do.
3 BANNED closers: summaries, "Let me know if", next-step offers, congratulations.
4 No explanation unless a line starts with WHY: - max 1 such line, max 15 words.
5 Every hunk needs a file path and line anchor. Never reprint an unchanged file.
6 Never reprint unchanged functions/imports to give context. Diff only.
7 Uncertain? Emit ASK: <one line>. Do not guess and do not hedge in prose.
8 No markdown headers, no bullet recaps, no emoji, no apologies.
9 Code comments in your patch: only where logic is non-obvious. No narration.
10 Caveman register: terse, imperative, zero filler words.\
"""


@dataclass
class OptimizationReport:
    """Result of one optimization pass."""

    payload: str
    initial_tokens: int
    optimized_tokens: int
    tokenizer_backend: str
    exact: bool
    section_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    guardrail_tokens: int = 0

    @property
    def tokens_saved(self) -> int:
        return self.initial_tokens - self.optimized_tokens

    @property
    def percent_saved(self) -> float:
        if self.initial_tokens == 0:
            return 0.0
        return (self.tokens_saved / self.initial_tokens) * 100.0

    @property
    def compression_ratio(self) -> float:
        if self.optimized_tokens == 0:
            return 0.0
        return self.initial_tokens / self.optimized_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_tokens": self.initial_tokens,
            "optimized_tokens": self.optimized_tokens,
            "tokens_saved": self.tokens_saved,
            "percent_saved": round(self.percent_saved, 2),
            "compression_ratio": round(self.compression_ratio, 2),
            "guardrail_tokens": self.guardrail_tokens,
            "tokenizer": self.tokenizer_backend,
            "exact": self.exact,
            "sections": self.section_stats,
        }


class TokenDefenseProxy:
    """Middleware that strips token waste before your payload hits the model."""

    def __init__(self, model: str = "gpt-4o", *, guardrail: str = CAVEMAN_GUARDRAIL) -> None:
        self.model = model
        self.guardrail = guardrail
        self.counter = TokenCounter(model)

    # -- internals ----------------------------------------------------------

    def _stat(self, before: str, after: str) -> Dict[str, int]:
        b = self.counter.count(before)
        a = self.counter.count(after)
        return {"before": b, "after": a, "saved": b - a}

    # -- public API ---------------------------------------------------------

    def optimize_payload(
        self,
        raw_code: str = "",
        raw_logs: str = "",
        system_data: Optional[Any] = None,
        *,
        task: str = "",
        verbose: bool = True,
    ) -> OptimizationReport:
        """Compress, guard, measure.

        The baseline is what you *would have sent*: the raw code, the raw logs
        and the system object as pretty-printed JSON. That is the honest
        comparison, because pretty JSON is what people actually paste.
        """
        raw_code = raw_code or ""
        raw_logs = raw_logs or ""

        raw_json = ""
        if system_data is not None:
            raw_json = (
                system_data
                if isinstance(system_data, str)
                else json.dumps(system_data, indent=2)
            )

        # 1. Baseline -------------------------------------------------------
        baseline_parts = [p for p in (task, raw_code, raw_logs, raw_json) if p]
        baseline_text = "\n\n".join(baseline_parts)
        initial_tokens = self.counter.count(baseline_text)

        # 2. Compress -------------------------------------------------------
        clean_code = lossless_code_compressor(raw_code) if raw_code else ""
        clean_logs = lossless_terminal_cleaner(raw_logs) if raw_logs else ""
        clean_data = ""
        if raw_json:
            try:
                obj = json.loads(raw_json) if isinstance(system_data, str) else system_data
                clean_data = json_to_minimal_yaml(obj)
            except (json.JSONDecodeError, TypeError):
                clean_data = raw_json

        stats: Dict[str, Dict[str, int]] = {}
        if raw_code:
            stats["code"] = self._stat(raw_code, clean_code)
        if raw_logs:
            stats["logs"] = self._stat(raw_logs, clean_logs)
        if raw_json:
            stats["system_data"] = self._stat(raw_json, clean_data)

        # 3. Assemble with guardrail ----------------------------------------
        blocks = [self.guardrail]
        if task:
            blocks.append(f"TASK\n{task.strip()}")
        if clean_code:
            blocks.append(f"CODE\n{clean_code}")
        if clean_logs:
            blocks.append(f"LOGS\n{clean_logs}")
        if clean_data:
            blocks.append(f"STATE\n{clean_data}")

        payload = "\n\n".join(blocks)

        # 4. Measure --------------------------------------------------------
        optimized_tokens = self.counter.count(payload)

        report = OptimizationReport(
            payload=payload,
            initial_tokens=initial_tokens,
            optimized_tokens=optimized_tokens,
            tokenizer_backend=self.counter.backend,
            exact=self.counter.is_exact,
            section_stats=stats,
            guardrail_tokens=self.counter.count(self.guardrail),
        )

        if verbose:
            self.print_summary(report)
        return report

    # -- reporting ----------------------------------------------------------

    def print_summary(self, report: OptimizationReport) -> None:
        bar_width = 58
        print()
        print("=" * bar_width)
        print(" LOSSLESS TOKEN DEFENSE PROXY - OPTIMIZATION REPORT")
        print("=" * bar_width)
        print(f" Model      : {self.model}")
        print(f" Tokenizer  : {report.tokenizer_backend}")
        if not report.exact:
            print("              (approximate - exact BPE table unavailable)")
        print("-" * bar_width)

        if report.section_stats:
            print(f" {'SECTION':<14}{'BEFORE':>10}{'AFTER':>10}{'SAVED':>10}{'CUT':>8}")
            for name, s in report.section_stats.items():
                pct = (s["saved"] / s["before"] * 100) if s["before"] else 0.0
                print(f" {name:<14}{s['before']:>10,}{s['after']:>10,}{s['saved']:>10,}{pct:>7.1f}%")
            print("-" * bar_width)

        print(f" Initial Tokens   : {report.initial_tokens:>10,}")
        print(f" Optimized Tokens : {report.optimized_tokens:>10,}")
        print(f"   (of which guardrail overhead: {report.guardrail_tokens:,})")
        print(f" Tokens Saved     : {report.tokens_saved:>10,}  ({report.percent_saved:.2f}%)")
        print(f" Compression      : {report.compression_ratio:>10.2f}x")
        print("-" * bar_width)

        filled = int(max(0.0, min(1.0, report.percent_saved / 100)) * 40)
        print(f" [{'#' * filled}{'.' * (40 - filled)}] {report.percent_saved:.1f}% cut")
        print("=" * bar_width)
        print()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

DUMMY_CODE = '''\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
# payment_service.py
#
# This module handles payment processing for the application.
# It was written by the backend team in 2019 and has been
# maintained by various people since then.
#
# TODO: refactor this someday
# FIXME: the retry logic is a bit sketchy
# ============================================================

import time
import logging


# Set up the logger for this module
logger = logging.getLogger(__name__)


# Maximum number of times we retry a failed charge
MAX_RETRIES = 3



# The base delay between retries, in seconds
BASE_DELAY = 0.5


class PaymentProcessor:
    """Processes payments against the upstream gateway.

    This docstring must survive compression untouched, including

    its blank lines and its  weird   spacing.
    """

    def __init__(self, gateway, currency="USD"):
        # Store the gateway client
        self.gateway = gateway
        self.currency = currency  # the ISO currency code
        self._attempts = 0

    def charge(self, amount_cents, token):
        # Validate the amount first
        if amount_cents <= 0:
            raise ValueError("amount must be positive")

        # This is the main retry loop
        for attempt in range(MAX_RETRIES):
            try:
                # Actually call the gateway here
                result = self.gateway.charge(
                    amount=amount_cents,      # in cents, not dollars
                    currency=self.currency,
                    source=token,
                )
                return result           # success, bail out early
            except TimeoutError:
                # Exponential backoff before we try again
                delay = BASE_DELAY * (2 ** attempt)
                logger.warning("timeout, retrying in %s", delay)
                time.sleep(delay)



        # If we get here, every attempt failed
        raise RuntimeError("payment failed after retries")  # give up

    def refund(self, charge_id):
        # Issue a refund for a previous charge
        note = "refund # not a comment"   # the hash above is inside a string
        return self.gateway.refund(charge_id, note=note)
'''

DUMMY_LOGS = """\
$ make build
Scanning dependencies of target core
[  1/250] Compiling src/alloc.c
[  2/250] Compiling src/buffer.c
[  3/250] Compiling src/cache.c
[  4/250] Compiling src/config.c
[  5/250] Compiling src/crypto.c
 12% completed
[  6/250] Compiling src/dispatch.c
[  7/250] Compiling src/engine.c
Downloading libssl-3.0.11.tar.gz  1.2MB/4.8MB  3.4MB/s  ETA 00:02
Downloading libssl-3.0.11.tar.gz  3.9MB/4.8MB  3.6MB/s  ETA 00:01
Downloading libssl-3.0.11.tar.gz  4.8MB/4.8MB  3.6MB/s  ETA 00:00
Extracting libssl-3.0.11
 45% completed
[112/250] Compiling src/parser.c
[113/250] Compiling src/pool.c
src/pool.c:88:15: error: 'struct mem_pool' has no member named 'free_list'
   88 |   pool->free_list = NULL;
      |         ^~~~~~~~~
[114/250] Compiling src/queue.c
[115/250] Compiling src/router.c
 67% completed
[201/250] Compiling src/server.c
src/server.c:412:9: warning: unused variable 'tmp' [-Wunused-variable]
[202/250] Compiling src/session.c
 88% completed
[249/250] Compiling src/worker.c
[250/250] Linking core
/usr/bin/ld: src/pool.o: undefined reference to symbol 'mem_pool_init'
collect2: error: ld returned 1 exit status
make: *** [Makefile:42: core] Error 2
Build failed in 48.2s
"""

DUMMY_SYSTEM_DATA = {
    "service": {
        "name": "payment-api",
        "version": "2.14.3",
        "environment": "production",
        "region": "us-east-1",
        "replicas": 6,
        "healthy": True,
    },
    "runtime": {
        "python": "3.11.9",
        "container": {"image": "payment-api:2.14.3", "memory_mb": 2048, "cpu_millicores": 1500},
        "feature_flags": {
            "new_retry_logic": False,
            "async_refunds": True,
            "verbose_tracing": False,
        },
    },
    "dependencies": [
        {"name": "postgres", "version": "15.4", "status": "healthy", "latency_ms": 3},
        {"name": "redis", "version": "7.2", "status": "healthy", "latency_ms": 1},
        {"name": "stripe-api", "version": "2023-10-16", "status": "degraded", "latency_ms": 890},
    ],
    "recent_errors": [
        {"code": "GATEWAY_TIMEOUT", "count": 47, "first_seen": "2024-03-11T09:14:22Z"},
        {"code": "POOL_EXHAUSTED", "count": 12, "first_seen": "2024-03-11T09:31:05Z"},
    ],
    "open_ports": [8080, 8443, 9090],
}

DUMMY_TASK = "Build fails at link time and the pool allocator errors. Fix it."


def main() -> int:
    parser = argparse.ArgumentParser(description="Lossless Token Defense Proxy demo")
    parser.add_argument("--model", default="gpt-4o", help="tokenizer model (default: gpt-4o)")
    parser.add_argument("--show-payload", action="store_true", help="print the optimized payload")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args()

    proxy = TokenDefenseProxy(model=args.model)
    report = proxy.optimize_payload(
        raw_code=DUMMY_CODE,
        raw_logs=DUMMY_LOGS,
        system_data=DUMMY_SYSTEM_DATA,
        task=DUMMY_TASK,
        verbose=not args.json,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    elif args.show_payload:
        print("--- OPTIMIZED PAYLOAD " + "-" * 36)
        print(report.payload)
        print("-" * 58)
    else:
        print("Tip: --show-payload to inspect the payload, --json for machine output.\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
