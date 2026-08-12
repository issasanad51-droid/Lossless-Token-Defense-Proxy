//! Lossless Token Defense Proxy - Rust port.
//!
//! Same guarantees as the Python reference implementation:
//! comments and progress noise are removed, meaning is not.

pub mod compressors;
pub mod data_converter;
pub mod orchestrator;
pub mod tokenizer;

pub use compressors::{lossless_code_compressor, lossless_terminal_cleaner, CodeOptions};
pub use data_converter::{json_to_minimal_yaml, minimal_yaml_to_json};
pub use orchestrator::{OptimizationReport, TokenDefenseProxy, CAVEMAN_GUARDRAIL};
pub use tokenizer::TokenCounter;
