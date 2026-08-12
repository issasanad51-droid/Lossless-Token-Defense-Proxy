//! Runnable demo: `cargo run --release`

use serde_json::json;
use token_defense_proxy::TokenDefenseProxy;

const DUMMY_CODE: &str = r#"#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
# payment_service.py
#
# This module handles payment processing for the application.
# It was written by the backend team in 2019.
#
# TODO: refactor this someday
# ============================================================

import time
import logging


# Set up the logger for this module
logger = logging.getLogger(__name__)


# Maximum number of times we retry a failed charge
MAX_RETRIES = 3



BASE_DELAY = 0.5


class PaymentProcessor:
    """Processes payments against the upstream gateway.

    This docstring must survive compression untouched.
    """

    def __init__(self, gateway, currency="USD"):
        # Store the gateway client
        self.gateway = gateway
        self.currency = currency  # the ISO currency code

    def charge(self, amount_cents, token):
        # Validate the amount first
        if amount_cents <= 0:
            raise ValueError("amount must be positive")

        # This is the main retry loop
        for attempt in range(MAX_RETRIES):
            try:
                result = self.gateway.charge(
                    amount=amount_cents,      # in cents, not dollars
                    currency=self.currency,
                    source=token,
                )
                return result           # success
            except TimeoutError:
                delay = BASE_DELAY * (2 ** attempt)
                time.sleep(delay)



        raise RuntimeError("payment failed after retries")  # give up

    def refund(self, charge_id):
        note = "refund # not a comment"   # hash above is inside a string
        return self.gateway.refund(charge_id, note=note)
"#;

const DUMMY_LOGS: &str = r#"$ make build
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
"#;

fn main() {
    let system_data = json!({
        "service": {
            "name": "payment-api",
            "version": "2.14.3",
            "environment": "production",
            "region": "us-east-1",
            "replicas": 6,
            "healthy": true
        },
        "runtime": {
            "python": "3.11.9",
            "container": {"image": "payment-api:2.14.3", "memory_mb": 2048, "cpu_millicores": 1500},
            "feature_flags": {"new_retry_logic": false, "async_refunds": true, "verbose_tracing": false}
        },
        "dependencies": [
            {"name": "postgres", "version": "15.4", "status": "healthy", "latency_ms": 3},
            {"name": "redis", "version": "7.2", "status": "healthy", "latency_ms": 1},
            {"name": "stripe-api", "version": "2023-10-16", "status": "degraded", "latency_ms": 890}
        ],
        "recent_errors": [
            {"code": "GATEWAY_TIMEOUT", "count": 47, "first_seen": "2024-03-11T09:14:22Z"},
            {"code": "POOL_EXHAUSTED", "count": 12, "first_seen": "2024-03-11T09:31:05Z"}
        ],
        "open_ports": [8080, 8443, 9090]
    });

    let show_payload = std::env::args().any(|a| a == "--show-payload");

    let proxy = TokenDefenseProxy::new("gpt-4o");
    let report = proxy.optimize_payload(
        DUMMY_CODE,
        DUMMY_LOGS,
        Some(&system_data),
        "Build fails at link time and the pool allocator errors. Fix it.",
    );

    proxy.print_summary(&report);

    if show_payload {
        println!("--- OPTIMIZED PAYLOAD {}", "-".repeat(36));
        println!("{}", report.payload);
        println!("{}", "-".repeat(58));
    } else {
        println!("Tip: run with --show-payload to inspect the payload.\n");
    }
}
