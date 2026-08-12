//! Lossless Token Defense Proxy Engine — central orchestration.
//!
//! Builds the Pipeline-Filter chain, drives messy dummy payloads through
//! the lossless optimizers, indexes the surviving code with the
//! Tree-Sitter boundary mapper + Personalized PageRank call-chain graph,
//! fuses Vector / AST / Regex lists with confidence-calibrated RRF, and
//! prints a tiktoken-rs telemetry summary.
//!
//! `cargo run --release` is the supported entry point (also the CI job).
//! The process always runs an in-binary self-test harness before the demo
//! so a red CI log cannot hide a broken filter.

mod filters;
mod indexer;
mod pipeline;

use std::process::ExitCode;
use std::time::Instant;

use filters::{baseline_pipeline, CodeCleaner, DataConverter, LogCleaner, PayloadKind};
use indexer::{RetrievalEngine, RrfBlender, CALLS_FUNCTION, HAS_PROPERTY, IMPORTS_MODULE};
use pipeline::{IdentityFilter, Pipeline, TokenFilter};
use tiktoken_rs::cl100k_base_singleton;

fn main() -> ExitCode {
    println!("{}", BANNER);

    let failed = run_self_tests();
    if failed > 0 {
        eprintln!("self-test harness: {failed} check(s) failed");
        return ExitCode::from(1);
    }
    println!("self-test harness: all checks passed\n");

    if let Err(err) = run_demo() {
        eprintln!("engine error: {err}");
        return ExitCode::from(2);
    }
    ExitCode::SUCCESS
}

const BANNER: &str = r#"
╔══════════════════════════════════════════════════════════════════╗
║           LOSSLESS TOKEN DEFENSE PROXY ENGINE                    ║
║   Pipeline-Filter · AST Scope Map · Weighted PPR · RRF Fusion    ║
╚══════════════════════════════════════════════════════════════════╝
"#;

fn count_tokens(text: &str) -> usize {
    cl100k_base_singleton()
        .encode_with_special_tokens(text)
        .len()
}

struct Payload {
    name: &'static str,
    kind: PayloadKind,
    body: &'static str,
}

fn demo_payloads() -> Vec<Payload> {
    vec![
        Payload {
            name: "auth_service.py",
            kind: PayloadKind::Code,
            body: MESSY_PYTHON,
        },
        Payload {
            name: "token_guard.rs",
            kind: PayloadKind::Code,
            body: MESSY_RUST,
        },
        Payload {
            name: "cargo-build.log",
            kind: PayloadKind::Log,
            body: MESSY_LOG,
        },
        Payload {
            name: "runtime-config.json",
            kind: PayloadKind::Json,
            body: NESTED_JSON,
        },
    ]
}

fn run_demo() -> Result<(), String> {
    let started = Instant::now();
    let code = CodeCleaner::new();
    let logs = LogCleaner::new();
    let data = DataConverter::new();

    let mut initial_parts = Vec::new();
    let mut final_parts = Vec::new();
    let mut cleaned_code: Vec<(String, String)> = Vec::new();

    println!("── stage 1 · lossless filters ──────────────────────────────");
    let mixed_pipeline = baseline_pipeline();
    println!(
        "  baseline pipeline ({} stages): {:?}",
        mixed_pipeline.len(),
        mixed_pipeline.stage_names()
    );
    for payload in demo_payloads() {
        let classified = PayloadKind::classify(payload.body);
        let filter: &dyn TokenFilter = match payload.kind {
            PayloadKind::Code => &code,
            PayloadKind::Log => &logs,
            PayloadKind::Json => &data,
        };
        let before_tok = count_tokens(payload.body);
        let cleaned = filter.filter(payload.body);
        let after_tok = count_tokens(&cleaned);
        let saved = before_tok.saturating_sub(after_tok);
        let pct = if before_tok == 0 {
            0.0
        } else {
            100.0 * saved as f64 / before_tok as f64
        };
        println!(
            "  {:<22} kind={:?} (classified={:?})  filter={:<16}  {:>5} → {:>5} tok   saved {:>5}  ({:>5.1}%)",
            payload.name,
            payload.kind,
            classified,
            filter.name(),
            before_tok,
            after_tok,
            saved,
            pct
        );
        initial_parts.push(payload.body.to_string());
        final_parts.push(cleaned.clone());
        if payload.kind == PayloadKind::Code {
            cleaned_code.push((payload.name.to_string(), cleaned));
        }
    }

    let initial = initial_parts.join("\n\n");
    let optimized = final_parts.join("\n\n");
    let initial_tokens = count_tokens(&initial);
    let final_tokens = count_tokens(&optimized);
    let saved = initial_tokens.saturating_sub(final_tokens);
    let pct = if initial_tokens == 0 {
        0.0
    } else {
        100.0 * saved as f64 / initial_tokens as f64
    };

    println!("\n── stage 2 · AST scope map + PPR + RRF ─────────────────────");
    let mut engine = RetrievalEngine::new();
    engine.blender = RrfBlender::new().with_k(60.0);
    for (path, src) in &cleaned_code {
        engine.index_file(path, src);
    }
    engine.finalize();
    {
        let runtime = engine.parser.runtime();
        let _ = runtime.language();
    }
    println!(
        "  indexed {} scope chunks · {} graph nodes · {} weighted edges",
        engine.chunks.len(),
        engine.graph.node_count(),
        engine.graph.edge_count()
    );
    if let Some((path, src)) = cleaned_code.first() {
        let idx = indexer::ast_parser::LineIndex::new(src);
        println!(
            "  {}  lines={}  bytes={}",
            path,
            idx.line_count(),
            idx.source_len()
        );
    }
    if let Some(sample) = engine.chunks.first() {
        println!(
            "  sample chunk {}  bytes={}..{}  ts-range={}:{}-{}:{}",
            sample.id(),
            sample.byte_start,
            sample.byte_end,
            sample.start_point().row,
            sample.start_point().column,
            sample.end_point().row,
            sample.end_point().column
        );
    }
    println!(
        "  edge weights: {}={:.1}  {}={:.1}  {}={:.1}",
        indexer::EdgeKind::CallsFunction.as_str(),
        CALLS_FUNCTION,
        indexer::EdgeKind::ImportsModule.as_str(),
        IMPORTS_MODULE,
        indexer::EdgeKind::HasProperty.as_str(),
        HAS_PROPERTY
    );
    if let Some(edge) = engine.graph.edges.first() {
        let from = &engine.graph.nodes[edge.from];
        let to = &engine.graph.nodes[edge.to];
        println!(
            "  sample edge: {} node#{} {} ({}) {}:{}-{} -> {}:{}  w={:.1}",
            edge.kind.as_str(),
            from.id,
            from.file_path,
            from.kind.as_str(),
            from.name,
            from.line_start,
            from.line_end,
            to.name,
            to.line_start,
            edge.weight
        );
    }

    let query = "authenticate user token jwt session";
    let outcome = engine.query(query, 6);
    println!("  query: {:?}", outcome.query);
    println!(
        "  rankers: vector={}  ast={}  regex={}  ppr_mass_nodes={}  graph={}/{}",
        outcome.vector.items.len(),
        outcome.ast.items.len(),
        outcome.regex.items.len(),
        outcome.ppr.len(),
        outcome.graph_nodes,
        outcome.graph_edges
    );
    for (i, cand) in outcome.blended.iter().enumerate() {
        let contrib = cand
            .contributions
            .iter()
            .map(|c| {
                format!(
                    "{}#{}:w{:.2}/c{:.2}={:.4}",
                    c.ranker, c.rank, c.weight, c.confidence, c.partial
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "  [{i}] {file}:{lo}-{hi}  score={score:.5}  conf={conf:.3}{flag}  members={}  [{contrib}]",
            cand.members.len(),
            file = cand.file_path,
            lo = cand.line_start,
            hi = cand.line_end,
            score = cand.score,
            conf = cand.confidence,
            flag = if cand.override_floated {
                "  OVERRIDE"
            } else {
                ""
            }
        );
    }

    let retrieved = outcome.context();
    let retrieved_tokens = count_tokens(&retrieved);

    let elapsed_ms = started.elapsed().as_millis();
    print_telemetry(initial_tokens, final_tokens, saved, pct, retrieved_tokens, elapsed_ms);
    Ok(())
}

fn print_telemetry(
    initial: usize,
    final_sz: usize,
    saved: usize,
    pct: f64,
    retrieved: usize,
    elapsed_ms: u128,
) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                  TELEMETRY SUMMARY  (cl100k_base)                ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Initial Token Size .............. {:>8}                     ║",
        initial
    );
    println!(
        "║  Final Optimized Token Size ...... {:>8}                     ║",
        final_sz
    );
    println!(
        "║  Total Lossless Tokens Saved ..... {:>8}                     ║",
        saved
    );
    println!(
        "║  Saving Percentage ............... {:>7.2}%                     ║",
        pct
    );
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Retrieved context tokens (RRF) .. {:>8}                     ║",
        retrieved
    );
    println!(
        "║  Wall time ....................... {:>7} ms                    ║",
        elapsed_ms
    );
    println!("╚══════════════════════════════════════════════════════════════════╝");
}

// ── in-binary test runner ────────────────────────────────────────────────

struct Check {
    name: &'static str,
    ok: bool,
    detail: String,
}

fn run_self_tests() -> usize {
    println!("── self-test runner ──────────────────────────────────────────────");
    let checks = collect_checks();
    let mut failed = 0usize;
    for c in &checks {
        if c.ok {
            println!("  PASS  {}", c.name);
        } else {
            println!("  FAIL  {}  — {}", c.name, c.detail);
            failed += 1;
        }
    }
    println!("  {} checks, {} failed", checks.len(), failed);
    failed
}

fn collect_checks() -> Vec<Check> {
    let mut out = Vec::new();

    // Pipeline wiring.
    {
        let mut p = Pipeline::new();
        p.register(IdentityFilter).register(CodeCleaner::new());
        let trace = p.execute_traced("let x = 1; // gone\n");
        out.push(Check {
            name: "pipeline_registers_code_cleaner",
            ok: p.len() == 2 && !p.is_empty() && p.stage_names()[1] == "code_cleaner",
            detail: format!("{:?}", p.stage_names()),
        });
        out.push(Check {
            name: "pipeline_execute_traced_saves_bytes",
            ok: trace.output_bytes() < trace.input_bytes()
                && trace.stages.iter().any(|s| s.bytes_saved() > 0),
            detail: format!(
                "{}→{} stages={} first={} us={}",
                trace.input_bytes(),
                trace.output_bytes(),
                trace.stages.len(),
                trace.stages.first().map(|s| s.name).unwrap_or("-"),
                trace.stages.iter().map(|s| s.elapsed_us).sum::<u128>()
            ),
        });
        let _ = p.execute("ok");
    }

    // Code cleaner: URLs, strings, attributes, indentation, shebang.
    {
        let c = CodeCleaner::new();
        let got = c.filter("let u = \"http://x.com\"; // trail\n");
        out.push(Check {
            name: "code_cleaner_keeps_url_strips_comment",
            ok: got.contains("http://x.com") && !got.contains("trail"),
            detail: got.clone(),
        });
        let got = c.filter("x = \"hello # world\"  # bye\n");
        out.push(Check {
            name: "code_cleaner_keeps_hash_in_string",
            ok: got.contains("hello # world") && !got.contains("bye"),
            detail: got,
        });
        let got = c.filter("#[derive(Debug)]\nstruct Foo;\n");
        out.push(Check {
            name: "code_cleaner_keeps_rust_attribute",
            ok: got.contains("#[derive(Debug)]"),
            detail: got,
        });
        let got = c.filter("def f():\n    return 1  # x\n");
        out.push(Check {
            name: "code_cleaner_preserves_indent",
            ok: got.contains("    return 1"),
            detail: got,
        });
        let got = c.filter("#!/usr/bin/env python3\n# c\nx=1\n");
        out.push(Check {
            name: "code_cleaner_preserves_shebang",
            ok: got.starts_with("#!/usr/bin/env python3"),
            detail: got,
        });
        let got = c.filter("fn foo<'a>(x: &'a str) { x } // z\n");
        out.push(Check {
            name: "code_cleaner_preserves_lifetimes",
            ok: got.contains("fn foo<'a>(x: &'a str)") && !got.contains("// z"),
            detail: got,
        });
    }

    // Log cleaner.
    {
        let l = LogCleaner::new();
        let got = l.filter("[1/500] step one\n[2/500] step two\nerror: boom\n");
        out.push(Check {
            name: "log_cleaner_erases_fraction_tickers",
            ok: !got.contains("[1/500]") && !got.contains("[2/500]") && got.contains("error: boom"),
            detail: got,
        });
        let got = l.filter("\u{1b}[31merror\u{1b}[0m: nope\n");
        out.push(Check {
            name: "log_cleaner_strips_ansi",
            ok: got.contains("error: nope") && !got.contains('\u{1b}'),
            detail: got,
        });
    }

    // JSON → YAML.
    {
        let d = DataConverter::new();
        let yaml = d.convert(r#"{"user":{"name":"alice","age":30},"ok":true}"#);
        out.push(Check {
            name: "data_converter_quote_free_yaml",
            ok: yaml.contains("name: alice")
                && yaml.contains("age: 30")
                && !yaml.contains('{')
                && !yaml.contains('"'),
            detail: yaml.clone(),
        });
        let json = r#"{"a":{"b":[1,2,{"c":"x"}]},"d":null}"#;
        match d.yaml_to_json(&d.convert(json)) {
            Ok(back) => {
                let ok = filters::data_converter::parse_json(json).ok()
                    == filters::data_converter::parse_json(&back).ok();
                out.push(Check {
                    name: "data_converter_round_trip",
                    ok,
                    detail: back,
                });
            }
            Err(e) => out.push(Check {
                name: "data_converter_round_trip",
                ok: false,
                detail: e,
            }),
        }
    }

    // AST chunks are structural, not windows.
    {
        let src = "fn a() {\n    let x = 1;\n}\nfn b() {\n    let y = 2;\n}\n";
        let chunks = indexer::AstParser::new().parse_file("t.rs", src);
        let names: Vec<_> = chunks.iter().map(|c| c.name.as_str()).collect();
        out.push(Check {
            name: "ast_parser_isolates_functions",
            ok: names.contains(&"a") && names.contains(&"b") && chunks.len() >= 2,
            detail: format!("{names:?}"),
        });
        if chunks.len() >= 2 {
            out.push(Check {
                name: "ast_parser_no_window_slice",
                ok: chunks[0].source.contains("fn a") && !chunks[0].source.contains("fn b"),
                detail: chunks[0].source.clone(),
            });
        }
    }

    // PPR surfaces callees of a focused node.
    {
        let src = "fn authenticate(t: &str) -> bool { validate_token(t) }\nfn validate_token(t: &str) -> bool { true }\nfn unrelated() { 1; }\n";
        let chunks = indexer::AstParser::new().parse_file("g.rs", src);
        let mut g = indexer::PprGraph::new();
        g.ingest_chunks(&chunks);
        let mut sources = std::collections::HashMap::new();
        sources.insert("g.rs".into(), src.to_string());
        g.wire_language_edges(&chunks, &sources);
        let auth = chunks.iter().find(|c| c.name == "authenticate");
        let ok = if let Some(auth) = auth {
            if let Some(id) = g.id_of(&auth.id()) {
                let mut focus = std::collections::HashMap::new();
                focus.insert(id, 1.0);
                let ranked = g.personalized_pagerank(&focus, 0.85, 30, 1e-9);
                let val = ranked
                    .iter()
                    .find(|(i, _)| g.nodes[*i].name == "validate_token")
                    .map(|(_, s)| *s)
                    .unwrap_or(0.0);
                let unr = ranked
                    .iter()
                    .find(|(i, _)| g.nodes[*i].name == "unrelated")
                    .map(|(_, s)| *s)
                    .unwrap_or(0.0);
                val >= unr
            } else {
                false
            }
        } else {
            false
        };
        out.push(Check {
            name: "ppr_prefers_callees_of_focus",
            ok,
            detail: format!("nodes={} edges={}", g.node_count(), g.edge_count()),
        });
    }

    // RRF merge + override.
    {
        use indexer::{RankedItem, RankedList, RrfBlender};
        let item = |id: &str, file: &str, s: usize, e: usize, r: usize, c: f64| RankedItem {
            id: id.into(),
            file_path: file.into(),
            line_start: s,
            line_end: e,
            content: id.into(),
            rank: r,
            confidence: c,
            source_ranker: String::new(),
        };
        let v = RankedList::new("vector", 1.0, vec![item("a", "f.rs", 10, 12, 1, 0.4)]);
        let a = RankedList::new("ast", 1.0, vec![item("b", "f.rs", 14, 18, 1, 0.4)]);
        let merged = RrfBlender::new().blend(&[v, a], &std::collections::HashMap::new());
        out.push(Check {
            name: "rrf_merges_15_line_proximity",
            ok: merged.len() == 1 && merged[0].line_start == 10 && merged[0].line_end == 18,
            detail: format!("n={} {:?}", merged.len(), merged.first().map(|c| (c.line_start, c.line_end))),
        });
        let v = RankedList::new("vector", 1.0, vec![item("hot", "a.rs", 1, 1, 9, 0.99)]);
        let a = RankedList::new("ast", 1.0, vec![item("cold", "b.rs", 1, 1, 1, 0.2)]);
        let fused = RrfBlender::new().blend(&[v, a], &std::collections::HashMap::new());
        out.push(Check {
            name: "rrf_confidence_override_floats",
            ok: !fused.is_empty() && fused[0].id == "hot" && fused[0].override_floated,
            detail: format!("{:?}", fused.first().map(|c| (c.id.clone(), c.override_floated))),
        });
    }

    // tiktoken is live.
    {
        let n = count_tokens("hello world");
        out.push(Check {
            name: "tiktoken_cl100k_live",
            ok: n > 0 && n < 10,
            detail: format!("{n}"),
        });
    }

    out
}

// ── dummy payloads ───────────────────────────────────────────────────────

const MESSY_PYTHON: &str = r#"#!/usr/bin/env python3
# auth_service.py — messy production leftover with comments everywhere
"""Authenticate inbound bearer tokens against the session store."""

import json   # stdlib
import time    # used for expiry
from hashlib import sha256   # hashing helper

# TODO: delete this once we migrate off the legacy store
LEGACY_SALT = "not-a-real-salt"  # noqa: S105


class AuthError(Exception):
    """Raised when a token cannot be trusted."""
    pass  # keep the class body so callers can except AuthError


class UserSession:
    # in-memory stand-in for redis
    def __init__(self, user_id, token, expires_at):
        self.user_id = user_id      # int
        self.token = token          # raw jwt
        self.expires_at = expires_at  # unix seconds

    def is_fresh(self):
        # compare against wall clock
        return time.time() < self.expires_at  # True if still valid


class AuthService:
    """Top-level façade used by the HTTP layer."""

    def __init__(self, store):
        self.store = store  # mapping of token -> UserSession

    def authenticate(self, bearer):
        # Strip the "Bearer " prefix if the caller sent one.
        token = self.normalize(bearer)  # may raise
        if not self.validate_token(token):
            raise AuthError("denied")  # do not leak why
        session = self.lookup_user(token)
        if session is None or not session.is_fresh():
            raise AuthError("expired")
        return session

    def normalize(self, bearer):
        # comments on every line on purpose
        if bearer is None:  # defensive
            raise AuthError("empty")
        text = bearer.strip()  # whitespace
        if text.lower().startswith("bearer "):  # RFC 6750
            text = text[7:]
        return text

    def validate_token(self, token):
        # structural checks only — crypto lives in decode_jwt
        if not token or len(token) < 8:  # too short
            return False
        return self.decode_jwt(token) is not None

    def decode_jwt(self, token):
        # fake decoder: split and hash, no real crypto
        parts = token.split(".")  # header.payload.sig
        if len(parts) != 3:  # classic jwt shape
            return None
        digest = sha256(token.encode("utf-8")).hexdigest()  # stand-in
        return {"sub": digest[:12], "raw": token}

    def lookup_user(self, token):
        # store is a dict in the dummy harness
        return self.store.get(token)  # may be None


def build_default_store():
    # seed a single session so the demo graph has data
    token = "aaa.bbb.ccc"  # dummy jwt
    sess = UserSession(42, token, time.time() + 3600)  # one hour
    return {token: sess}
"#;

const MESSY_RUST: &str = r#"
//! token_guard.rs — intentionally comment-heavy so the cleaner has work.

use std::collections::HashMap; // session table
use std::time::{SystemTime, UNIX_EPOCH}; // expiry

/// Recoverable auth failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthError {
    Denied,   // bad signature / shape
    Expired,  // wall clock
    Empty,    // missing header
}

/// In-memory stand-in for a session row.
#[derive(Debug, Clone)]
pub struct UserSession {
    pub user_id: u64,      // primary key
    pub token: String,     // raw jwt
    pub expires_at: u64,   // unix seconds
}

impl UserSession {
    pub fn is_fresh(&self) -> bool {
        // compare against wall clock
        now_secs() < self.expires_at
    }
}

/// HTTP-facing façade. Mirrors the Python AuthService so the PPR graph
/// can resolve cross-language call names in the demo.
pub struct TokenGuard {
    store: HashMap<String, UserSession>, // token -> session
}

impl TokenGuard {
    pub fn new(store: HashMap<String, UserSession>) -> Self {
        Self { store } // move
    }

    pub fn authenticate(&self, bearer: &str) -> Result<UserSession, AuthError> {
        let token = self.normalize(bearer)?; // may fail
        if !self.validate_token(&token) {
            return Err(AuthError::Denied); // do not leak why
        }
        let session = self.lookup_user(&token).ok_or(AuthError::Denied)?;
        if !session.is_fresh() {
            return Err(AuthError::Expired);
        }
        Ok(session)
    }

    pub fn normalize(&self, bearer: &str) -> Result<String, AuthError> {
        let text = bearer.trim(); // whitespace
        if text.is_empty() {
            return Err(AuthError::Empty);
        }
        let lower = text.to_ascii_lowercase();
        if lower.starts_with("bearer ") {
            return Ok(text[7..].to_string()); // RFC 6750
        }
        Ok(text.to_string())
    }

    pub fn validate_token(&self, token: &str) -> bool {
        // structural checks only
        if token.len() < 8 {
            return false;
        }
        self.decode_jwt(token).is_some()
    }

    pub fn decode_jwt(&self, token: &str) -> Option<JwtClaims> {
        let mut parts = token.split('.'); // header.payload.sig
        let _h = parts.next()?;
        let _p = parts.next()?;
        let _s = parts.next()?;
        if parts.next().is_some() {
            return None; // too many dots
        }
        Some(JwtClaims {
            sub: token.len() as u64, // stand-in
        })
    }

    pub fn lookup_user(&self, token: &str) -> Option<UserSession> {
        self.store.get(token).cloned() // owned copy for the caller
    }
}

/// Tiny claims struct so HAS_PROPERTY has something to hang on.
#[derive(Debug, Clone)]
pub struct JwtClaims {
    pub sub: u64, // subject
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
"#;

const MESSY_LOG: &str = "\
\u{1b}[1m\u{1b}[32m   Compiling\u{1b}[0m lossless-token-defense-proxy v1.0.0
\u{1b}[1m\u{1b}[32m   Compiling\u{1b}[0m regex v1.10.6
\u{1b}[1m\u{1b}[32m   Compiling\u{1b}[0m aho-corasick v1.1.3
\u{1b}[1m\u{1b}[32m   Compiling\u{1b}[0m memchr v2.7.4
\u{1b}[1m\u{1b}[32m   Compiling\u{1b}[0m tiktoken-rs v0.12.0
\u{1b}[1m\u{1b}[32m   Compiling\u{1b}[0m tree-sitter v0.20.10
\u{1b}[1m\u{1b}[32m   Compiling\u{1b}[0m fancy-regex v0.13.0
\u{1b}[1m\u{1b}[32m   Compiling\u{1b}[0m bstr v1.10.0
   Compiling cfg-if v1.0.0
   Compiling libc v0.2.155
   Compiling once_cell v1.19.0
   Compiling serde v1.0.203
[1/500] Building token_guard.rs
[2/500] Building auth_service.py
[3/500] Linking ltdp
[4/500] Linking ltdp
[5/500] Linking ltdp
progress 1%\rprogress 14%\rprogress 37%\rprogress 61%\rprogress 88%\rprogress 100%
[====================>               ] 45%
[===========================>        ] 72%
[====================================] 100%
|/-\\|/-\\|/-\\ waiting for file lock on package cache
   Compiling unicode-ident v1.0.12
   Compiling proc-macro2 v1.0.86
   Compiling quote v1.0.36
   Compiling syn v2.0.68
   Compiling serde_derive v1.0.203
   Compiling thiserror v1.0.61
   Compiling thiserror-impl v1.0.61
   Compiling anyhow v1.0.86
Downloading crates ... 3.1 MiB/s ETA 00:12
Downloaded regex v1.10.6
    Finished `release` profile [optimized] target(s) in 12.44s
error[E0308]: mismatched types
  --> src/token_guard.rs:88:9
   |
88 |     Ok(session)
   |     ^^^^^^^^^^^ expected Result, found UserSession
warning: unused import: `std::fs`
  --> src/main.rs:4:5
error: could not compile `ltdp` (bin \"ltdp\") due to 1 previous error
";

const NESTED_JSON: &str = r#"
{
  "service": {
    "name": "lossless-token-defense-proxy",
    "version": "1.0.0",
    "environment": "production",
    "features": {
      "code_cleaner": true,
      "log_cleaner": true,
      "data_converter": true,
      "ppr_graph": true,
      "rrf_blender": true
    }
  },
  "retrieval": {
    "window_lines": 15,
    "rrf_k": 60,
    "confidence_override": 0.95,
    "weights": {
      "vector": 1.15,
      "ast": 1.0,
      "regex": 0.85
    },
    "edge_weights": {
      "CALLS_FUNCTION": 3.0,
      "IMPORTS_MODULE": 2.0,
      "HAS_PROPERTY": 1.0
    }
  },
  "tenants": [
    {
      "id": "alpha",
      "region": "us-east-1",
      "quota_tokens": 128000,
      "flags": {
        "beta_rrf": true,
        "note": "15.4"
      }
    },
    {
      "id": "bravo",
      "region": "eu-west-1",
      "quota_tokens": 64000,
      "flags": {
        "beta_rrf": false,
        "note": "true"
      }
    }
  ],
  "null_slot": null
}
"#;
