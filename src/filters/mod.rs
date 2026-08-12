//! Layer 1 & Layer 2 baseline filters.
//!
//! * Layer 1 — [`code_cleaner::CodeCleaner`]: structural comment / whitespace
//!   defense for source payloads.
//! * Layer 1 — [`log_cleaner::LogCleaner`]: ticker / ANSI / progress defense
//!   for terminal build logs.
//! * Layer 2 — [`data_converter::DataConverter`]: lossless JSON → quote-free
//!   YAML rewrite for configuration payloads.
//!
//! [`baseline_pipeline`] registers both layers in that order so a mixed
//! orchestrator can hand any payload to a single chain. The orchestrator in
//! `main.rs` still routes by content kind so a JSON document is never run
//! through the comment stripper first.

pub mod code_cleaner;
pub mod data_converter;
pub mod log_cleaner;

pub use code_cleaner::CodeCleaner;
pub use data_converter::DataConverter;
pub use log_cleaner::LogCleaner;

use crate::pipeline::Pipeline;

/// Register the Layer 1 + Layer 2 baseline filters on a fresh pipeline.
pub fn baseline_pipeline() -> Pipeline {
    let mut pipeline = Pipeline::new();
    pipeline
        .register(CodeCleaner::new())
        .register(LogCleaner::new())
        .register(DataConverter::new());
    pipeline
}

/// Best-effort payload classifier used by the orchestrator to pick a filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadKind {
    Code,
    Log,
    Json,
}

impl PayloadKind {
    pub fn classify(input: &str) -> Self {
        let trimmed = input.trim_start();
        if looks_like_json(trimmed) {
            return PayloadKind::Json;
        }
        if looks_like_log(input) {
            return PayloadKind::Log;
        }
        PayloadKind::Code
    }
}

fn looks_like_json(trimmed: &str) -> bool {
    let bytes = trimmed.as_bytes();
    if bytes.is_empty() {
        return false;
    }
    let first = bytes[0];
    if first != b'{' && first != b'[' {
        return false;
    }
    // Cheap structural vote: require a closing twin and at least one colon
    // or a JSON literal keyword so we do not hijack code blocks that happen
    // to start with `{`.
    let last = trimmed.trim_end().as_bytes().last().copied();
    let closes = match first {
        b'{' => last == Some(b'}'),
        b'[' => last == Some(b']'),
        _ => false,
    };
    if !closes {
        return false;
    }
    trimmed.contains(':') || trimmed.contains("true") || trimmed.contains("null")
}

fn looks_like_log(input: &str) -> bool {
    let mut score = 0i32;
    if input.contains('\r') {
        score += 2;
    }
    if input.contains("\u{1b}[") {
        score += 2;
    }
    if input.contains("[1/") || input.contains("[ 1/") {
        score += 2;
    }
    let mut ticker_hits = 0;
    for line in input.lines().take(80) {
        let t = line.trim();
        if t.starts_with("Compiling ")
            || t.starts_with("Checking ")
            || t.starts_with("Downloading ")
            || t.contains("% |")
            || t.contains("ETA")
            || (t.starts_with('[') && t.contains('/'))
        {
            ticker_hits += 1;
        }
    }
    score += (ticker_hits / 2) as i32;
    score >= 3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_json_object() {
        assert_eq!(
            PayloadKind::classify("{\"a\": 1}\n"),
            PayloadKind::Json
        );
    }

    #[test]
    fn classifies_rust_as_code() {
        assert_eq!(
            PayloadKind::classify("fn main() {\n    println!(\"hi\");\n}\n"),
            PayloadKind::Code
        );
    }
}
