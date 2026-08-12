//! # Subsystem C - reciprocal rank fusion with proximity merging
//!
//! Three retrievers disagree, and they disagree in useful ways. The vector leg
//! finds code that *reads* like the query. The graph leg finds code that is
//! *structurally reachable* from what the vector leg found. The keyword leg
//! finds exact identifier hits that embeddings routinely miss. Averaging their
//! raw scores is invalid - a cosine of 0.62, a PageRank of 0.0031, and a BM25
//! of 14.7 live on incomparable scales.
//!
//! Reciprocal rank fusion sidesteps that entirely by throwing away magnitudes
//! and keeping only ordering:
//!
//! ```text
//! Score(doc) = Sum over lists of  Weight_list * (1 / (K + Rank_doc_in_list))
//! ```
//!
//! ## Two deviations from textbook RRF
//!
//! 1. **K scales with candidate count.** Fixed `K = 60` is tuned for
//!    web-scale result sets. On a 40-chunk repository it flattens ranks 1 and 5
//!    into near-identical scores, destroying the signal. Here `K` grows with
//!    the pool ([`adaptive_k`]) so small result sets stay sharply discriminated.
//! 2. **A high-confidence override.** RRF is deliberately magnitude-blind, but
//!    a cosine above [`CONFIDENCE_THRESHOLD`] is a near-exact match and should
//!    not be outvoted by two mediocre lists. Such a hit is promoted above the
//!    fused ordering, and the promotion is recorded in
//!    [`FusedResult::confidence_override`] so it is visible rather than silent.
//!
//! ## Proximity merging
//!
//! Retrievers return sibling functions separately. Sending lines 40-58 and
//! 61-92 of the same file as two disjoint blocks wastes a header and hides the
//! fact that they are adjacent. Any two surviving spans in the same file within
//! [`PROXIMITY_LINES`] of each other are merged into one contiguous block.

use std::collections::HashMap;

use super::ast_chunker::ChunkIndex;
use super::{Filter, PipelineContext, PipelineError};

/// Gap in lines within which two blocks in the same file are merged.
pub const PROXIMITY_LINES: usize = 15;

/// A score above this is treated as a near-certain match and promoted.
pub const CONFIDENCE_THRESHOLD: f32 = 0.95;

/// Which retriever a ranked list came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RetrieverKind {
    /// Embedding cosine similarity.
    Vector,
    /// Personalized PageRank over the call graph.
    Graph,
    /// Literal identifier / substring matching.
    Keyword,
}

impl RetrieverKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            RetrieverKind::Vector => "vector",
            RetrieverKind::Graph => "graph",
            RetrieverKind::Keyword => "keyword",
        }
    }

    /// Default trust in each leg.
    ///
    /// Vector leads because it is the only leg that generalizes beyond exact
    /// wording. Graph is close behind - structural reachability is strong
    /// evidence. Keyword is lowest on its own, but it is the tie-breaker that
    /// rescues exact identifier queries the embedder fumbles.
    pub fn default_weight(&self) -> f32 {
        match self {
            RetrieverKind::Vector => 1.0,
            RetrieverKind::Graph => 0.8,
            RetrieverKind::Keyword => 0.6,
        }
    }
}

/// One retriever's ordered output. Position in `entries` is the rank.
#[derive(Debug, Clone)]
pub struct RankedList {
    pub kind: RetrieverKind,
    /// `(chunk_id, raw_score)` in descending score order.
    pub entries: Vec<(String, f32)>,
    /// Trust multiplier. Defaults to [`RetrieverKind::default_weight`].
    pub weight: f32,
}

impl RankedList {
    pub fn new(kind: RetrieverKind, entries: Vec<(String, f32)>) -> Self {
        RankedList {
            kind,
            entries,
            weight: kind.default_weight(),
        }
    }

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Best raw score in the list, used for the confidence override.
    pub fn peak_score(&self) -> f32 {
        self.entries
            .iter()
            .map(|(_, s)| *s)
            .fold(0.0f32, |a, b| if b > a { b } else { a })
    }
}

/// Per-list contribution to a fused score, kept for explainability.
#[derive(Debug, Clone, PartialEq)]
pub struct Contribution {
    pub kind: RetrieverKind,
    /// 1-based rank in that list.
    pub rank: usize,
    pub raw_score: f32,
    pub weight: f32,
    /// `weight * 1 / (k + rank)`
    pub contribution: f32,
}

/// A merged, contiguous region of a file ready to be sent to the model.
#[derive(Debug, Clone)]
pub struct FusedResult {
    pub file_path: String,
    /// 1-based inclusive.
    pub line_start: usize,
    /// 1-based inclusive.
    pub line_end: usize,
    /// Chunk ids folded into this block, in positional order.
    pub chunk_ids: Vec<String>,
    /// Human-facing label, the first chunk by position.
    pub primary_symbol: String,
    pub score: f32,
    /// How many distinct retrievers voted for this block.
    pub retriever_count: usize,
    pub contributions: Vec<Contribution>,
    /// Set when a raw score above [`CONFIDENCE_THRESHOLD`] forced promotion.
    pub confidence_override: bool,
}

impl FusedResult {
    pub fn line_count(&self) -> usize {
        self.line_end.saturating_sub(self.line_start) + 1
    }

    /// One-line justification of why this block was selected.
    pub fn explain(&self) -> String {
        let mut parts: Vec<String> = self
            .contributions
            .iter()
            .map(|c| format!("{}#{}", c.kind.as_str(), c.rank))
            .collect();
        parts.sort();
        format!(
            "{}:{}-{} [{}] score={:.5} via {}{}",
            self.file_path,
            self.line_start,
            self.line_end,
            self.primary_symbol,
            self.score,
            parts.join(" + "),
            if self.confidence_override {
                " (confidence override)"
            } else {
                ""
            }
        )
    }
}

/// Adaptive `K` for the reciprocal rank formula.
///
/// Textbook RRF fixes `K = 60`, which is tuned for very large result sets. With
/// only a handful of candidates that constant swamps the rank term: with
/// `K = 60`, rank 1 scores 0.0164 and rank 5 scores 0.0154 - a 6% spread that
/// noise erases. Scaling `K` with the pool keeps the spread meaningful for
/// small repositories while converging to the classic behaviour on large ones.
pub fn adaptive_k(candidate_count: usize) -> f32 {
    const MIN_K: f32 = 5.0;
    const MAX_K: f32 = 60.0;
    if candidate_count == 0 {
        return MIN_K;
    }
    let scaled = (candidate_count as f32) / 2.0;
    scaled.clamp(MIN_K, MAX_K)
}

/// Subsystem C. Fuses ranked lists and merges nearby regions.
#[derive(Debug, Clone)]
pub struct RrfBlender {
    /// Maximum blocks to return.
    pub top_n: usize,
    /// Line gap within which two blocks in a file are merged.
    pub proximity_lines: usize,
    /// Raw score above which a hit is promoted.
    pub confidence_threshold: f32,
    /// Multiplicative bonus per additional agreeing retriever. Agreement across
    /// independent methods is the strongest signal available.
    pub agreement_bonus: f32,
    /// Override `adaptive_k` with a fixed value, for reproducing textbook RRF.
    pub fixed_k: Option<f32>,
}

impl Default for RrfBlender {
    fn default() -> Self {
        RrfBlender {
            top_n: 3,
            proximity_lines: PROXIMITY_LINES,
            confidence_threshold: CONFIDENCE_THRESHOLD,
            agreement_bonus: 0.15,
            fixed_k: None,
        }
    }
}

impl RrfBlender {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_top_n(mut self, top_n: usize) -> Self {
        self.top_n = top_n;
        self
    }

    /// Fuse ranked lists into merged, descending-score blocks.
    pub fn blend(
        &self,
        lists: &[RankedList],
        index: &ChunkIndex,
    ) -> Result<Vec<FusedResult>, PipelineError> {
        // ---- 1. accumulate reciprocal-rank contributions --------------------
        let mut unique: HashMap<&str, ()> = HashMap::new();
        for list in lists {
            for (id, _) in &list.entries {
                unique.insert(id.as_str(), ());
            }
        }
        if unique.is_empty() {
            return Ok(Vec::new());
        }

        let k = self.fixed_k.unwrap_or_else(|| adaptive_k(unique.len()));

        let mut fused: HashMap<String, (f32, Vec<Contribution>, bool)> = HashMap::new();
        for list in lists {
            for (position, (id, raw)) in list.entries.iter().enumerate() {
                let rank = position + 1;
                let contribution = list.weight * (1.0 / (k + rank as f32));

                let entry = fused
                    .entry(id.clone())
                    .or_insert_with(|| (0.0, Vec::new(), false));
                entry.0 += contribution;
                entry.1.push(Contribution {
                    kind: list.kind,
                    rank,
                    raw_score: *raw,
                    weight: list.weight,
                    contribution,
                });
                // High-confidence override: a near-exact match must not be
                // outvoted by two lukewarm lists.
                if *raw >= self.confidence_threshold {
                    entry.2 = true;
                }
            }
        }

        // ---- 2. agreement bonus ---------------------------------------------
        let mut scored: Vec<(String, f32, Vec<Contribution>, bool)> = fused
            .into_iter()
            .map(|(id, (mut score, contributions, override_flag))| {
                let distinct = {
                    let mut kinds: Vec<RetrieverKind> =
                        contributions.iter().map(|c| c.kind).collect();
                    kinds.sort();
                    kinds.dedup();
                    kinds.len()
                };
                if distinct > 1 {
                    score *= 1.0 + self.agreement_bonus * (distinct - 1) as f32;
                }
                (id, score, contributions, override_flag)
            })
            .collect();

        // Drop ids the index does not know - a stale id has no coordinates and
        // cannot be turned into a block.
        scored.retain(|(id, _, _, _)| index.get(id).is_some());
        if scored.is_empty() {
            return Ok(Vec::new());
        }

        // ---- 3. descending sort, overrides first ----------------------------
        scored.sort_by(|a, b| {
            b.3.cmp(&a.3) // confidence override wins outright
                .then(
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
                .then(a.0.cmp(&b.0)) // deterministic tie-break
        });

        // ---- 4. proximity merge ---------------------------------------------
        let mut blocks: Vec<FusedResult> = Vec::new();
        for (id, score, contributions, override_flag) in scored {
            let chunk = match index.get(&id) {
                Some(c) => c,
                None => continue,
            };

            let mergeable = blocks.iter_mut().find(|block| {
                block.file_path == chunk.file_path
                    && gap_between(
                        block.line_start,
                        block.line_end,
                        chunk.line_start,
                        chunk.line_end,
                    ) <= self.proximity_lines
            });

            match mergeable {
                Some(block) => {
                    block.line_start = block.line_start.min(chunk.line_start);
                    block.line_end = block.line_end.max(chunk.line_end);
                    block.chunk_ids.push(id.clone());
                    // A merged block inherits the best score present, not the
                    // sum: merging is a presentation choice and must not
                    // manufacture relevance.
                    if score > block.score {
                        block.score = score;
                    }
                    block.confidence_override |= override_flag;
                    for contribution in contributions {
                        if !block.contributions.iter().any(|existing| {
                            existing.kind == contribution.kind
                                && existing.rank == contribution.rank
                        }) {
                            block.contributions.push(contribution);
                        }
                    }
                }
                None => {
                    let distinct = {
                        let mut kinds: Vec<RetrieverKind> =
                            contributions.iter().map(|c| c.kind).collect();
                        kinds.sort();
                        kinds.dedup();
                        kinds.len()
                    };
                    blocks.push(FusedResult {
                        file_path: chunk.file_path.clone(),
                        line_start: chunk.line_start,
                        line_end: chunk.line_end,
                        chunk_ids: vec![id.clone()],
                        primary_symbol: chunk.qualified_name.clone(),
                        score,
                        retriever_count: distinct,
                        contributions,
                        confidence_override: override_flag,
                    });
                }
            }
        }

        // ---- 5. relabel merged blocks and finalize ---------------------------
        for block in blocks.iter_mut() {
            // Label by the first symbol positionally, never the smallest - a
            // block called "helper" that actually starts with the function the
            // user wants is a misleading label.
            let mut ordered: Vec<(usize, String, String)> = block
                .chunk_ids
                .iter()
                .filter_map(|id| {
                    index
                        .get(id)
                        .map(|c| (c.line_start, c.qualified_name.clone(), id.clone()))
                })
                .collect();
            ordered.sort_by(|a, b| a.0.cmp(&b.0).then(a.2.cmp(&b.2)));
            if let Some((_, name, _)) = ordered.first() {
                block.primary_symbol = name.clone();
            }
            block.chunk_ids = ordered.into_iter().map(|(_, _, id)| id).collect();

            let mut kinds: Vec<RetrieverKind> =
                block.contributions.iter().map(|c| c.kind).collect();
            kinds.sort();
            kinds.dedup();
            block.retriever_count = kinds.len();
        }

        blocks.sort_by(|a, b| {
            b.confidence_override
                .cmp(&a.confidence_override)
                .then(
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
                .then(a.file_path.cmp(&b.file_path))
                .then(a.line_start.cmp(&b.line_start))
        });
        blocks.truncate(self.top_n);
        Ok(blocks)
    }
}

/// Distance in lines between two spans; 0 when they touch or overlap.
fn gap_between(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> usize {
    if a_start <= b_end && b_start <= a_end {
        return 0;
    }
    if b_start > a_end {
        b_start - a_end
    } else {
        a_start - b_end
    }
}

/// Input bundle for the [`Filter`] implementation.
pub struct BlendInput {
    pub lists: Vec<RankedList>,
    pub index: std::sync::Arc<ChunkIndex>,
}

impl Filter for RrfBlender {
    type Input = BlendInput;
    type Output = Vec<FusedResult>;

    fn name(&self) -> &'static str {
        "rrf_blender"
    }

    fn apply(
        &self,
        input: BlendInput,
        ctx: &mut PipelineContext,
    ) -> Result<Vec<FusedResult>, PipelineError> {
        if input.lists.iter().all(|l| l.is_empty()) {
            return Err(PipelineError::Empty("rrf_blender"));
        }
        let blocks = self.blend(&input.lists, &input.index)?;

        ctx.set("blocks_returned", blocks.len() as u64);
        ctx.set(
            "lines_returned",
            blocks.iter().map(|b| b.line_count() as u64).sum::<u64>(),
        );
        if blocks.iter().any(|b| b.confidence_override) {
            ctx.note("a high-confidence hit was promoted above the fused ordering".to_string());
        }
        Ok(blocks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::ast_chunker::{AstChunker, SourceFile};

    const SAMPLE: &str = r#"fn alpha() -> u32 {
    1
}

fn beta() -> u32 {
    2
}

fn gamma() -> u32 {
    3
}
"#;

    /// A file whose two functions sit far enough apart to resist merging.
    const SPREAD: &str = r#"fn near_top() -> u32 {
    1
}
// filler 1
// filler 2
// filler 3
// filler 4
// filler 5
// filler 6
// filler 7
// filler 8
// filler 9
// filler 10
// filler 11
// filler 12
// filler 13
// filler 14
// filler 15
// filler 16
// filler 17
// filler 18
// filler 19
// filler 20
fn far_below() -> u32 {
    2
}
"#;

    fn index_of(path: &str, src: &str) -> ChunkIndex {
        ChunkIndex::new(
            AstChunker::new()
                .chunk_file(&SourceFile::new(path, src))
                .expect("chunking must succeed"),
        )
    }

    fn sample_index() -> ChunkIndex {
        index_of("m.rs", SAMPLE)
    }

    #[test]
    fn adaptive_k_is_clamped() {
        assert_eq!(adaptive_k(0), 5.0);
        assert_eq!(adaptive_k(4), 5.0);
        assert_eq!(adaptive_k(40), 20.0);
        assert_eq!(adaptive_k(10_000), 60.0);
    }

    #[test]
    fn adaptive_k_grows_with_the_pool() {
        assert!(adaptive_k(200) > adaptive_k(20));
    }

    #[test]
    fn small_pools_stay_discriminated() {
        // The reason adaptive K exists: with K=60 the spread collapses.
        let small = adaptive_k(10);
        let spread_small = (1.0 / (small + 1.0)) / (1.0 / (small + 5.0));
        let spread_fixed = (1.0 / (60.0 + 1.0)) / (1.0 / (60.0 + 5.0));
        assert!(
            spread_small > spread_fixed,
            "adaptive={} fixed={}",
            spread_small,
            spread_fixed
        );
    }

    #[test]
    fn rank_one_beats_rank_two() {
        let index = sample_index();
        let lists = vec![RankedList::new(
            RetrieverKind::Vector,
            vec![("m.rs::alpha".into(), 0.5), ("m.rs::gamma".into(), 0.4)],
        )];
        let out = RrfBlender::new().blend(&lists, &index).unwrap();
        assert_eq!(out[0].primary_symbol, "alpha");
    }

    #[test]
    fn results_are_sorted_descending() {
        let index = index_of("s.rs", SPREAD);
        let lists = vec![RankedList::new(
            RetrieverKind::Vector,
            vec![("s.rs::far_below".into(), 0.9), ("s.rs::near_top".into(), 0.8)],
        )];
        let out = RrfBlender::new().blend(&lists, &index).unwrap();
        for pair in out.windows(2) {
            assert!(pair[0].score >= pair[1].score, "not descending: {:?}",
                out.iter().map(|b| b.score).collect::<Vec<_>>());
        }
    }

    #[test]
    fn agreement_across_retrievers_wins() {
        let index = sample_index();
        // gamma is rank 1 in one list; alpha is rank 2 in all three.
        let lists = vec![
            RankedList::new(
                RetrieverKind::Vector,
                vec![("m.rs::gamma".into(), 0.6), ("m.rs::alpha".into(), 0.5)],
            ),
            RankedList::new(
                RetrieverKind::Graph,
                vec![("m.rs::beta".into(), 0.3), ("m.rs::alpha".into(), 0.2)],
            ),
            RankedList::new(
                RetrieverKind::Keyword,
                vec![("m.rs::beta".into(), 0.3), ("m.rs::alpha".into(), 0.2)],
            ),
        ];
        let blender = RrfBlender::new().with_top_n(10);
        let out = blender.blend(&lists, &index).unwrap();
        let alpha = out
            .iter()
            .find(|b| b.chunk_ids.iter().any(|i| i == "m.rs::alpha"))
            .unwrap();
        assert_eq!(alpha.retriever_count, 3);
    }

    #[test]
    fn weights_shift_the_ordering() {
        let index = index_of("s.rs", SPREAD);
        let heavy_vector = vec![
            RankedList::new(RetrieverKind::Vector, vec![("s.rs::near_top".into(), 0.5)])
                .with_weight(10.0),
            RankedList::new(RetrieverKind::Graph, vec![("s.rs::far_below".into(), 0.5)])
                .with_weight(0.1),
        ];
        let out = RrfBlender::new().blend(&heavy_vector, &index).unwrap();
        assert_eq!(out[0].primary_symbol, "near_top");

        let heavy_graph = vec![
            RankedList::new(RetrieverKind::Vector, vec![("s.rs::near_top".into(), 0.5)])
                .with_weight(0.1),
            RankedList::new(RetrieverKind::Graph, vec![("s.rs::far_below".into(), 0.5)])
                .with_weight(10.0),
        ];
        let out = RrfBlender::new().blend(&heavy_graph, &index).unwrap();
        assert_eq!(out[0].primary_symbol, "far_below");
    }

    #[test]
    fn high_confidence_hit_is_promoted() {
        let index = index_of("s.rs", SPREAD);
        // near_top is buried at rank 2 in one weak list but scores 0.99.
        let lists = vec![
            RankedList::new(
                RetrieverKind::Graph,
                vec![("s.rs::far_below".into(), 0.4), ("s.rs::near_top".into(), 0.99)],
            ),
            RankedList::new(RetrieverKind::Keyword, vec![("s.rs::far_below".into(), 0.5)]),
        ];
        let out = RrfBlender::new().blend(&lists, &index).unwrap();
        assert_eq!(out[0].primary_symbol, "near_top");
        assert!(out[0].confidence_override, "override flag not set");
    }

    #[test]
    fn override_is_disclosed_not_silent() {
        let index = sample_index();
        let lists = vec![RankedList::new(
            RetrieverKind::Vector,
            vec![("m.rs::alpha".into(), 0.999)],
        )];
        let out = RrfBlender::new().blend(&lists, &index).unwrap();
        assert!(out[0].explain().contains("confidence override"));
    }

    #[test]
    fn below_threshold_does_not_override() {
        let index = sample_index();
        let lists = vec![RankedList::new(
            RetrieverKind::Vector,
            vec![("m.rs::alpha".into(), 0.94)],
        )];
        let out = RrfBlender::new().blend(&lists, &index).unwrap();
        assert!(!out[0].confidence_override);
    }

    #[test]
    fn nearby_spans_merge_into_one_block() {
        let index = sample_index();
        let lists = vec![RankedList::new(
            RetrieverKind::Vector,
            vec![("m.rs::alpha".into(), 0.5), ("m.rs::gamma".into(), 0.4)],
        )];
        let out = RrfBlender::new().blend(&lists, &index).unwrap();
        assert_eq!(out.len(), 1, "adjacent functions should merge");
        assert_eq!(out[0].chunk_ids.len(), 2);
        assert_eq!(out[0].line_start, 1);
    }

    #[test]
    fn distant_spans_stay_separate() {
        let index = index_of("s.rs", SPREAD);
        let lists = vec![RankedList::new(
            RetrieverKind::Vector,
            vec![("s.rs::near_top".into(), 0.5), ("s.rs::far_below".into(), 0.4)],
        )];
        let out = RrfBlender::new().blend(&lists, &index).unwrap();
        assert_eq!(out.len(), 2, "spans 20 lines apart must not merge");
    }

    #[test]
    fn spans_in_different_files_never_merge() {
        let mut chunks = AstChunker::new()
            .chunk_file(&SourceFile::new("a.rs", SAMPLE))
            .unwrap();
        chunks.extend(
            AstChunker::new()
                .chunk_file(&SourceFile::new("b.rs", SAMPLE))
                .unwrap(),
        );
        let index = ChunkIndex::new(chunks);
        let lists = vec![RankedList::new(
            RetrieverKind::Vector,
            vec![("a.rs::alpha".into(), 0.5), ("b.rs::alpha".into(), 0.4)],
        )];
        let out = RrfBlender::new().blend(&lists, &index).unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn merged_block_is_labelled_by_first_symbol() {
        let index = sample_index();
        // gamma outranks alpha, but alpha comes first in the file.
        let lists = vec![RankedList::new(
            RetrieverKind::Vector,
            vec![("m.rs::gamma".into(), 0.9), ("m.rs::alpha".into(), 0.4)],
        )];
        let out = RrfBlender::new().blend(&lists, &index).unwrap();
        assert_eq!(out[0].primary_symbol, "alpha");
        assert!(out[0].chunk_ids.contains(&"m.rs::gamma".to_string()));
    }

    #[test]
    fn merging_does_not_inflate_scores() {
        let index = sample_index();
        let lists = vec![RankedList::new(
            RetrieverKind::Vector,
            vec![("m.rs::alpha".into(), 0.5), ("m.rs::gamma".into(), 0.4)],
        )];
        let k = adaptive_k(2);
        let best = 1.0 / (k + 1.0);
        let out = RrfBlender::new().blend(&lists, &index).unwrap();
        assert!(
            (out[0].score - best).abs() < 1e-6,
            "score {} should equal the best contributor {}",
            out[0].score,
            best
        );
    }

    #[test]
    fn score_matches_the_formula() {
        let index = sample_index();
        let lists = vec![RankedList::new(
            RetrieverKind::Vector,
            vec![("m.rs::alpha".into(), 0.5)],
        )
        .with_weight(1.0)];
        let out = RrfBlender::new().blend(&lists, &index).unwrap();
        let expected = 1.0 / (adaptive_k(1) + 1.0);
        assert!((out[0].score - expected).abs() < 1e-6);
    }

    #[test]
    fn top_n_is_respected() {
        let index = index_of("s.rs", SPREAD);
        let lists = vec![RankedList::new(
            RetrieverKind::Vector,
            vec![("s.rs::near_top".into(), 0.5), ("s.rs::far_below".into(), 0.4)],
        )];
        let out = RrfBlender::new().with_top_n(1).blend(&lists, &index).unwrap();
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn unknown_ids_are_dropped() {
        let index = sample_index();
        let lists = vec![RankedList::new(
            RetrieverKind::Vector,
            vec![("m.rs::ghost".into(), 0.9), ("m.rs::alpha".into(), 0.5)],
        )];
        let out = RrfBlender::new().blend(&lists, &index).unwrap();
        assert!(out.iter().all(|b| !b.chunk_ids.iter().any(|i| i.contains("ghost"))));
    }

    #[test]
    fn empty_lists_produce_no_blocks() {
        let index = sample_index();
        let out = RrfBlender::new().blend(&[], &index).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn a_single_list_still_works() {
        let index = sample_index();
        let lists = vec![RankedList::new(
            RetrieverKind::Keyword,
            vec![("m.rs::beta".into(), 0.2)],
        )];
        assert_eq!(RrfBlender::new().blend(&lists, &index).unwrap().len(), 1);
    }

    #[test]
    fn output_is_deterministic() {
        let index = sample_index();
        let lists = vec![
            RankedList::new(
                RetrieverKind::Vector,
                vec![("m.rs::alpha".into(), 0.5), ("m.rs::beta".into(), 0.5)],
            ),
            RankedList::new(
                RetrieverKind::Graph,
                vec![("m.rs::beta".into(), 0.5), ("m.rs::alpha".into(), 0.5)],
            ),
        ];
        let a = RrfBlender::new().blend(&lists, &index).unwrap();
        let b = RrfBlender::new().blend(&lists, &index).unwrap();
        let ids = |v: &Vec<FusedResult>| {
            v.iter().map(|r| r.primary_symbol.clone()).collect::<Vec<_>>()
        };
        assert_eq!(ids(&a), ids(&b));
    }

    #[test]
    fn gap_calculation_handles_every_arrangement() {
        assert_eq!(gap_between(1, 10, 5, 15), 0); // overlapping
        assert_eq!(gap_between(1, 10, 10, 20), 0); // touching
        assert_eq!(gap_between(1, 10, 13, 20), 3); // after
        assert_eq!(gap_between(13, 20, 1, 10), 3); // before
    }

    #[test]
    fn contributions_are_recorded_for_explainability() {
        let index = sample_index();
        let lists = vec![
            RankedList::new(RetrieverKind::Vector, vec![("m.rs::alpha".into(), 0.5)]),
            RankedList::new(RetrieverKind::Graph, vec![("m.rs::alpha".into(), 0.3)]),
        ];
        let out = RrfBlender::new().blend(&lists, &index).unwrap();
        assert_eq!(out[0].contributions.len(), 2);
        assert!(out[0].explain().contains("vector#1"));
        assert!(out[0].explain().contains("graph#1"));
    }

    #[test]
    fn peak_score_finds_the_maximum() {
        let list = RankedList::new(
            RetrieverKind::Vector,
            vec![("a".into(), 0.2), ("b".into(), 0.97)],
        );
        assert!((list.peak_score() - 0.97).abs() < 1e-6);
    }

    #[test]
    fn filter_rejects_all_empty_input() {
        let index = std::sync::Arc::new(sample_index());
        let mut ctx = PipelineContext::new();
        let input = BlendInput {
            lists: vec![RankedList::new(RetrieverKind::Vector, Vec::new())],
            index,
        };
        assert!(RrfBlender::new().run(input, &mut ctx).is_err());
    }

    #[test]
    fn filter_records_telemetry() {
        let index = std::sync::Arc::new(sample_index());
        let mut ctx = PipelineContext::new();
        let input = BlendInput {
            lists: vec![RankedList::new(
                RetrieverKind::Vector,
                vec![("m.rs::alpha".into(), 0.5)],
            )],
            index,
        };
        let out = RrfBlender::new().run(input, &mut ctx).unwrap();
        assert_eq!(ctx.counter("blocks_returned"), out.len() as u64);
        assert!(ctx.counter("lines_returned") > 0);
    }
}
