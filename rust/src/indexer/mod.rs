//! Bespoke retrieval ecosystem for the Lossless Token Defense Proxy.
//!
//! Three subsystems, each a [`Filter`], wired into one pipeline:
//!
//! * [`ast_chunker`]  - Subsystem A: AST-bounded vector chunking.
//! * [`ppr_graph`]    - Subsystem B: semantic-weighted personalized PageRank.
//! * [`rrf_blender`]  - Subsystem C: confidence-calibrated reciprocal rank fusion.
//!
//! # Why a trait-based pipeline
//!
//! The three stages need to hand each other *precise structural data*, not
//! reformatted text: the chunker's exact line coordinates flow into the graph,
//! the graph's node ids flow into the blender, and the blender's merged line
//! ranges flow back to the original source. Passing typed values through a
//! common [`Filter`] trait keeps those coordinates intact end to end - nothing
//! is ever re-derived from a string, which is where retrieval stacks normally
//! lose a line number and start emitting half a function.
//!
//! Each filter declares its own `Input`/`Output` associated types, so the
//! compiler enforces that the stages line up. [`PipelineContext`] carries
//! cross-cutting telemetry (timings, counters, notes) without polluting those
//! signatures.

pub mod ast_chunker;
pub mod ppr_graph;
pub mod rrf_blender;

use std::collections::BTreeMap;
use std::fmt;
use std::time::Instant;

/// Errors any stage can raise. Deliberately small and owned - these cross
/// thread boundaries in the parallel retrieval fan-out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineError {
    /// A source file could not be read.
    Io(String),
    /// A file was read but could not be scanned into scopes.
    Parse { file: String, reason: String },
    /// A stage received no usable input.
    Empty(&'static str),
    /// A caller supplied nonsensical configuration.
    Config(String),
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineError::Io(msg) => write!(f, "io error: {}", msg),
            PipelineError::Parse { file, reason } => {
                write!(f, "parse error in {}: {}", file, reason)
            }
            PipelineError::Empty(stage) => write!(f, "stage `{}` received no input", stage),
            PipelineError::Config(msg) => write!(f, "invalid configuration: {}", msg),
        }
    }
}

impl std::error::Error for PipelineError {}

/// Telemetry shared by every stage.
#[derive(Debug, Default, Clone)]
pub struct PipelineContext {
    /// Wall-clock microseconds per stage, keyed by [`Filter::name`].
    pub stage_micros: BTreeMap<String, u128>,
    /// Arbitrary integer counters (files scanned, edges built, ...).
    pub counters: BTreeMap<String, u64>,
    /// Human-readable diagnostics, surfaced in the CLI report.
    pub notes: Vec<String>,
}

impl PipelineContext {
    pub fn new() -> Self {
        PipelineContext::default()
    }

    pub fn record(&mut self, stage: &str, micros: u128) {
        let slot = self.stage_micros.entry(stage.to_string()).or_insert(0);
        *slot += micros;
    }

    pub fn bump(&mut self, key: &str, by: u64) {
        let slot = self.counters.entry(key.to_string()).or_insert(0);
        *slot += by;
    }

    pub fn set(&mut self, key: &str, value: u64) {
        self.counters.insert(key.to_string(), value);
    }

    pub fn note<S: Into<String>>(&mut self, msg: S) {
        self.notes.push(msg.into());
    }

    pub fn counter(&self, key: &str) -> u64 {
        *self.counters.get(key).unwrap_or(&0)
    }

    pub fn total_micros(&self) -> u128 {
        self.stage_micros.values().sum()
    }
}

/// One stage of the retrieval pipeline.
///
/// Implementors define [`Filter::apply`]; callers should invoke
/// [`Filter::run`], which times the stage into the [`PipelineContext`].
pub trait Filter {
    type Input;
    type Output;

    /// Stable identifier used as the telemetry key.
    fn name(&self) -> &'static str;

    /// The actual work.
    fn apply(
        &self,
        input: Self::Input,
        ctx: &mut PipelineContext,
    ) -> Result<Self::Output, PipelineError>;

    /// Timed wrapper around [`Filter::apply`]. Not intended to be overridden.
    fn run(
        &self,
        input: Self::Input,
        ctx: &mut PipelineContext,
    ) -> Result<Self::Output, PipelineError> {
        let started = Instant::now();
        let result = self.apply(input, ctx);
        ctx.record(self.name(), started.elapsed().as_micros());
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Doubler;

    impl Filter for Doubler {
        type Input = i64;
        type Output = i64;

        fn name(&self) -> &'static str {
            "doubler"
        }

        fn apply(&self, input: i64, ctx: &mut PipelineContext) -> Result<i64, PipelineError> {
            ctx.bump("doubled", 1);
            Ok(input * 2)
        }
    }

    #[test]
    fn run_times_the_stage_and_forwards_output() {
        let mut ctx = PipelineContext::new();
        let out = Doubler.run(21, &mut ctx).expect("doubler must succeed");
        assert_eq!(out, 42);
        assert_eq!(ctx.counter("doubled"), 1);
        assert!(ctx.stage_micros.contains_key("doubler"));
    }

    #[test]
    fn context_accumulates() {
        let mut ctx = PipelineContext::new();
        ctx.record("a", 10);
        ctx.record("a", 5);
        ctx.bump("n", 2);
        ctx.bump("n", 3);
        assert_eq!(ctx.stage_micros.get("a"), Some(&15));
        assert_eq!(ctx.counter("n"), 5);
        assert_eq!(ctx.total_micros(), 15);
    }

    #[test]
    fn errors_display_readably() {
        let err = PipelineError::Parse {
            file: "a.rs".to_string(),
            reason: "unbalanced".to_string(),
        };
        assert!(err.to_string().contains("a.rs"));
    }
}

// ---------------------------------------------------------------------------
// End-to-end orchestration
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::sync::Arc;

use self::ast_chunker::{AstChunker, ChunkIndex, SourceFile};
use self::ppr_graph::{CodeGraph, PprEngine, PprInput};
use self::rrf_blender::{BlendInput, FusedResult, RankedList, RetrieverKind, RrfBlender};

/// How many vector hits seed the random walk.
const SEED_COUNT: usize = 10;
/// How deep each retriever's list runs before fusion.
const CANDIDATE_DEPTH: usize = 25;
/// Below this a PPR score means "the walk never reached this node".
const PPR_SCORE_EPSILON: f64 = 1e-12;

/// A built index plus its graph, reusable across queries.
///
/// Chunking and graph construction are the expensive steps and depend only on
/// the source, so they happen once in [`RetrievalEngine::build`]; each query
/// then only pays for the three retrieval legs and fusion.
pub struct RetrievalEngine {
    pub index: Arc<ChunkIndex>,
    pub graph: Arc<CodeGraph>,
    pub blender: RrfBlender,
    pub ppr: PprEngine,
}

/// Everything one query produced, including the audit trail.
pub struct RetrievalOutcome {
    pub query: String,
    pub blocks: Vec<FusedResult>,
    pub ctx: PipelineContext,
}

impl RetrievalOutcome {
    /// Fraction of indexed lines that did **not** need to be sent.
    pub fn savings_ratio(&self, total_lines: usize) -> f64 {
        if total_lines == 0 {
            return 0.0;
        }
        let returned: usize = self.blocks.iter().map(|b| b.line_count()).sum();
        1.0 - (returned as f64 / total_lines as f64)
    }
}

impl RetrievalEngine {
    /// Run Subsystem A, then build the Subsystem B graph.
    pub fn build(files: Vec<SourceFile>) -> Result<(Self, PipelineContext), PipelineError> {
        let mut ctx = PipelineContext::new();
        let index = AstChunker::new().run(files, &mut ctx)?;
        if index.is_empty() {
            return Err(PipelineError::Empty("ast_chunker"));
        }

        let started = Instant::now();
        let graph = CodeGraph::from_chunk_index(&index);
        ctx.record("graph_build", started.elapsed().as_micros());
        ctx.set("graph_nodes", graph.node_count() as u64);
        ctx.set("graph_edges", graph.edge_count() as u64);

        Ok((
            RetrievalEngine {
                index: Arc::new(index),
                graph: Arc::new(graph),
                blender: RrfBlender::new(),
                ppr: PprEngine::new(),
            },
            ctx,
        ))
    }

    /// Literal identifier matching - the leg that rescues exact-name queries
    /// the embedder fumbles.
    fn keyword_search(&self, query: &str, top_k: usize) -> Vec<(String, f32)> {
        let needles: Vec<String> = query
            .split(|c: char| !(c.is_alphanumeric() || c == '_'))
            .filter(|t| t.len() > 2)
            .map(|t| t.to_lowercase())
            .collect();
        if needles.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(String, f32)> = self
            .index
            .chunks
            .iter()
            .filter_map(|chunk| {
                let name = chunk.qualified_name.to_lowercase();
                let signature = chunk.signature.to_lowercase();
                let mut score = 0.0f32;
                for needle in &needles {
                    // An exact symbol name is the strongest literal evidence
                    // available, so it saturates the confidence threshold.
                    if name == *needle {
                        score += 1.0;
                    } else if name.contains(needle.as_str()) {
                        score += 0.5;
                    } else if signature.contains(needle.as_str()) {
                        score += 0.2;
                    }
                }
                if score > 0.0 {
                    Some((chunk.id.clone(), score / needles.len() as f32))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(top_k);
        scored
    }

    /// Run all three legs and fuse them.
    pub fn query(&self, query: &str) -> Result<RetrievalOutcome, PipelineError> {
        let mut ctx = PipelineContext::new();
        if query.trim().is_empty() {
            return Err(PipelineError::Config("empty query".to_string()));
        }

        // Leg 1 - vector.
        let started = Instant::now();
        let vector_hits = self.index.vector_search(query, CANDIDATE_DEPTH);
        ctx.record("vector_search", started.elapsed().as_micros());
        ctx.set("vector_hits", vector_hits.len() as u64);

        // Leg 2 - graph, seeded by the best vector hits.
        let seeds: HashMap<String, f64> = vector_hits
            .iter()
            .take(SEED_COUNT)
            .map(|(id, score)| (id.clone(), *score as f64))
            .collect();
        let ppr_result = if seeds.is_empty() {
            Default::default()
        } else {
            self.ppr.run(
                PprInput::new(Arc::clone(&self.graph), seeds),
                &mut ctx,
            )?
        };
        // Drop zero-score nodes. PPR returns a score for *every* node, and the
        // ones unreachable from the seeds sit at exactly 0.0. Feeding those to
        // the blender hands them a real RRF rank purely by array position,
        // which drags unrelated files into the answer and destroys the savings.
        let graph_hits: Vec<(String, f32)> = ppr_result
            .top(CANDIDATE_DEPTH)
            .iter()
            .filter(|(_, score)| *score > PPR_SCORE_EPSILON)
            .map(|(id, score)| (id.clone(), *score as f32))
            .collect();
        ctx.set("graph_hits", graph_hits.len() as u64);

        // Leg 3 - keyword.
        let started = Instant::now();
        let keyword_hits = self.keyword_search(query, CANDIDATE_DEPTH);
        ctx.record("keyword_search", started.elapsed().as_micros());
        ctx.set("keyword_hits", keyword_hits.len() as u64);

        // Subsystem C.
        let lists = vec![
            RankedList::new(RetrieverKind::Vector, vector_hits),
            RankedList::new(RetrieverKind::Graph, graph_hits),
            RankedList::new(RetrieverKind::Keyword, keyword_hits),
        ];
        if lists.iter().all(|l| l.is_empty()) {
            return Ok(RetrievalOutcome {
                query: query.to_string(),
                blocks: Vec::new(),
                ctx,
            });
        }

        let blocks = self.blender.run(
            BlendInput {
                lists,
                index: Arc::clone(&self.index),
            },
            &mut ctx,
        )?;

        Ok(RetrievalOutcome {
            query: query.to_string(),
            blocks,
            ctx,
        })
    }

    pub fn total_lines(&self) -> usize {
        self.index.total_lines()
    }
}

#[cfg(test)]
mod pipeline_tests {
    use super::*;

    fn corpus() -> Vec<SourceFile> {
        vec![
            SourceFile::new(
                "auth.rs",
                "pub fn verify_token(token: &str) -> bool {\n    let parsed = decode_token(token);\n    parsed\n}\n\nfn decode_token(token: &str) -> bool {\n    !token.is_empty()\n}\n",
            ),
            SourceFile::new(
                "billing.py",
                "def charge_customer(account, cents):\n    total = apply_tax(cents)\n    return total\n\n\ndef apply_tax(cents):\n    return int(cents * 1.08)\n",
            ),
        ]
    }

    #[test]
    fn builds_an_engine_over_mixed_languages() {
        let (engine, ctx) = RetrievalEngine::build(corpus()).expect("build must succeed");
        assert!(engine.index.len() >= 4, "got {} chunks", engine.index.len());
        assert_eq!(ctx.counter("files_indexed"), 2);
        assert!(ctx.counter("graph_nodes") > 0);
    }

    #[test]
    fn end_to_end_query_returns_blocks() {
        let (engine, _) = RetrievalEngine::build(corpus()).unwrap();
        let outcome = engine.query("verify token").expect("query must succeed");
        assert!(!outcome.blocks.is_empty());
        assert!(outcome
            .blocks
            .iter()
            .any(|b| b.primary_symbol.contains("verify") || b.primary_symbol.contains("decode")));
    }

    #[test]
    fn query_returns_far_less_than_the_whole_corpus() {
        let (engine, _) = RetrievalEngine::build(corpus()).unwrap();
        let outcome = engine.query("apply tax to cents").unwrap();
        let ratio = outcome.savings_ratio(engine.total_lines());
        assert!(ratio > 0.0, "no lines were saved");
    }

    #[test]
    fn empty_query_is_rejected() {
        let (engine, _) = RetrievalEngine::build(corpus()).unwrap();
        assert!(engine.query("   ").is_err());
    }

    #[test]
    fn building_with_no_usable_files_errors() {
        let files = vec![SourceFile::new("notes.txt", "just prose")];
        assert!(RetrievalEngine::build(files).is_err());
    }

    #[test]
    fn keyword_leg_finds_exact_symbols() {
        let (engine, _) = RetrievalEngine::build(corpus()).unwrap();
        let hits = engine.keyword_search("decode_token", 10);
        assert!(
            hits.iter().any(|(id, _)| id.contains("decode_token")),
            "got {:?}",
            hits
        );
    }

    #[test]
    fn telemetry_covers_every_stage() {
        let (engine, _) = RetrievalEngine::build(corpus()).unwrap();
        let outcome = engine.query("charge customer").unwrap();
        assert!(outcome.ctx.stage_micros.contains_key("vector_search"));
        assert!(outcome.ctx.stage_micros.contains_key("keyword_search"));
        assert!(outcome.ctx.stage_micros.contains_key("rrf_blender"));
    }

    #[test]
    fn unrelated_files_stay_out_of_the_answer() {
        // Regression: PPR scores every node, and unreachable ones come back as
        // exactly 0.0. If those are passed to the blender they still earn an
        // RRF rank from their array position, so a query about billing would
        // pull in auth.rs and the savings ratio would collapse to zero.
        let (engine, _) = RetrievalEngine::build(corpus()).unwrap();
        let outcome = engine.query("charge customer").unwrap();
        assert!(
            outcome.blocks.iter().all(|b| b.file_path == "billing.py"),
            "unrelated file leaked in: {:?}",
            outcome
                .blocks
                .iter()
                .map(|b| b.file_path.clone())
                .collect::<Vec<_>>()
        );
        let ratio = outcome.savings_ratio(engine.total_lines());
        assert!(ratio > 0.25, "expected real savings, got {:.1}%", ratio * 100.0);
    }

    #[test]
    fn queries_are_deterministic() {
        let (engine, _) = RetrievalEngine::build(corpus()).unwrap();
        let a = engine.query("apply tax").unwrap();
        let b = engine.query("apply tax").unwrap();
        let names = |o: &RetrievalOutcome| {
            o.blocks.iter().map(|x| x.primary_symbol.clone()).collect::<Vec<_>>()
        };
        assert_eq!(names(&a), names(&b));
    }
}
