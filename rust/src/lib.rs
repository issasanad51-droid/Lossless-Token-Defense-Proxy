//! Lossless Token Defense Proxy - Rust port.
//!
//! Same guarantees as the Python reference implementation:
//! comments and progress noise are removed, meaning is not.

pub mod compressors;
pub mod data_converter;
pub mod indexer;
pub mod orchestrator;
pub mod tokenizer;

pub use compressors::{lossless_code_compressor, lossless_terminal_cleaner, CodeOptions};
pub use data_converter::{json_to_minimal_yaml, minimal_yaml_to_json};
pub use orchestrator::{OptimizationReport, TokenDefenseProxy, CAVEMAN_GUARDRAIL};
pub use tokenizer::TokenCounter;

pub use indexer::ast_chunker::{AstChunker, ChunkIndex, CodeChunk, Language, ScopeKind, SourceFile};
pub use indexer::ppr_graph::{CodeGraph, EdgeType, PprEngine, PprResult};
pub use indexer::rrf_blender::{FusedResult, RankedList, RetrieverKind, RrfBlender};
pub use indexer::{Filter, PipelineContext, PipelineError, RetrievalEngine, RetrievalOutcome};
