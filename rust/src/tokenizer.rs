//! Token counting via tiktoken-rs, with a graceful offline fallback.
//!
//! tiktoken-rs embeds its BPE tables in the binary, so unlike the Python
//! package it does not need network access at runtime.

use tiktoken_rs::{cl100k_base, o200k_base, CoreBPE};

pub struct TokenCounter {
    model: String,
    bpe: Option<CoreBPE>,
    backend: String,
    exact: bool,
}

impl TokenCounter {
    pub fn new(model: &str) -> Self {
        // o200k_base backs gpt-4o / gpt-4.1 / o-series; cl100k_base backs gpt-4 / gpt-3.5.
        let use_o200k = model.starts_with("gpt-4o")
            || model.starts_with("gpt-4.1")
            || model.starts_with("o1")
            || model.starts_with("o3")
            || model.starts_with("o4");

        let (bpe, backend, exact) = if use_o200k {
            match o200k_base() {
                Ok(b) => (Some(b), "tiktoken-rs:o200k_base".to_string(), true),
                Err(_) => match cl100k_base() {
                    Ok(b) => (Some(b), format!("tiktoken-rs:cl100k_base (proxy for {})", model), false),
                    Err(_) => (None, "heuristic".to_string(), false),
                },
            }
        } else {
            match cl100k_base() {
                Ok(b) => (Some(b), "tiktoken-rs:cl100k_base".to_string(), true),
                Err(_) => (None, "heuristic".to_string(), false),
            }
        };

        Self { model: model.to_string(), bpe, backend, exact }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn backend(&self) -> &str {
        &self.backend
    }

    pub fn is_exact(&self) -> bool {
        self.exact
    }

    pub fn count(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        match &self.bpe {
            Some(bpe) => bpe.encode_ordinary(text).len(),
            None => Self::estimate(text),
        }
    }

    /// Deterministic stand-in used only if no BPE table loads.
    fn estimate(text: &str) -> usize {
        let mut tokens = 0usize;
        for word in text.split_whitespace() {
            tokens += 1 + word.len() / 5;
        }
        tokens.max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_are_positive_and_scale() {
        let c = TokenCounter::new("gpt-4o");
        assert_eq!(c.count(""), 0);
        assert!(c.count("hello world") > 0);
        assert!(c.count("hello world hello world") > c.count("hello world"));
    }

    #[test]
    fn gpt4o_uses_o200k() {
        let c = TokenCounter::new("gpt-4o");
        assert!(c.backend().contains("o200k"), "backend was {}", c.backend());
        assert!(c.is_exact());
    }

    #[test]
    fn gpt4_uses_cl100k() {
        let c = TokenCounter::new("gpt-4");
        assert!(c.backend().contains("cl100k"), "backend was {}", c.backend());
    }
}
