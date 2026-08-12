//! # Subsystem B - semantic-weighted personalized PageRank
//!
//! A generic graph library treats every edge as equivalent. Code does not work
//! that way: "function A calls function B" is a far stronger statement about
//! what a reader needs to see next than "module A imports module B". So the
//! edge weights here are semantic and hardcoded by relationship type:
//!
//! | Relationship      | Weight | Why |
//! |-------------------|--------|-----|
//! | `CallsFunction`   | 3.0    | Direct execution dependency - almost always required context |
//! | `ImportsModule`   | 2.0    | Structural dependency - often required |
//! | `HasProperty`     | 1.0    | Containment - useful for orientation, rarely the answer |
//!
//! ## Personalized, not global
//!
//! Global PageRank answers "what is important in this codebase?" - and the
//! answer is always the same regardless of the query, which makes it useless
//! for retrieval. Personalized PageRank answers "what is important *given that
//! the user is looking at these nodes?*", by restarting the random walk at a
//! seed distribution taken from Subsystem A's top vector matches. Score then
//! ripples outward along call chains, so a helper three calls deep still
//! surfaces if it is on a hot path from the seed.
//!
//! Implemented as power iteration on a row-normalized transition matrix, in
//! plain Rust with no external graph crate. See [`PprConfig`] for why the
//! damping factor here is 0.6 rather than the textbook 0.85 - on a small,
//! bidirectionally-traversed code graph the classic value lets central nodes
//! outrank the seed, which defeats the point of personalization.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use super::ast_chunker::{ChunkIndex, ScopeKind};
use super::{Filter, PipelineContext, PipelineError};

/// Semantic relationship types, each with a fixed traversal weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EdgeType {
    /// One function invokes another.
    CallsFunction,
    /// A module or file depends on another module.
    ImportsModule,
    /// A container owns a member (struct field, class method).
    HasProperty,
}

impl EdgeType {
    /// Hardcoded directional execution weight.
    pub fn weight(&self) -> f64 {
        match self {
            EdgeType::CallsFunction => 3.0,
            EdgeType::ImportsModule => 2.0,
            EdgeType::HasProperty => 1.0,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            EdgeType::CallsFunction => "CALLS_FUNCTION",
            EdgeType::ImportsModule => "IMPORTS_MODULE",
            EdgeType::HasProperty => "HAS_PROPERTY",
        }
    }
}

/// A code element in the graph.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub id: String,
    pub name: String,
    /// Scope kind as a string, e.g. `function`, `struct`.
    pub type_variant: String,
    pub file_path: String,
    pub line_start: usize,
    pub line_end: usize,
}

/// A directed, semantically typed dependency.
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub source: String,
    pub target: String,
    pub edge_type: EdgeType,
    /// `edge_type.weight()` times any multiplicity observed.
    pub weight: f64,
}

/// Directed multigraph of code elements.
#[derive(Debug, Clone, Default)]
pub struct CodeGraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    index_of: HashMap<String, usize>,
    /// `source_index -> [(target_index, weight)]`
    adjacency: Vec<Vec<(usize, f64)>>,
    /// `target_index -> [(source_index, weight)]`
    reverse: Vec<Vec<(usize, f64)>>,
}

impl CodeGraph {
    pub fn new() -> Self {
        CodeGraph::default()
    }

    pub fn add_node(&mut self, node: Node) -> usize {
        if let Some(existing) = self.index_of.get(&node.id) {
            return *existing;
        }
        let index = self.nodes.len();
        self.index_of.insert(node.id.clone(), index);
        self.nodes.push(node);
        self.adjacency.push(Vec::new());
        self.reverse.push(Vec::new());
        index
    }

    /// Add an edge. Repeated identical edges accumulate weight rather than
    /// duplicating - calling a helper five times is a stronger dependency than
    /// calling it once.
    pub fn add_edge(&mut self, source: &str, target: &str, edge_type: EdgeType) {
        if source == target {
            return; // self-loops add nothing and distort normalization
        }
        let (si, ti) = match (self.index_of.get(source), self.index_of.get(target)) {
            (Some(s), Some(t)) => (*s, *t),
            _ => return, // never invent nodes from a dangling reference
        };

        let weight = edge_type.weight();
        if let Some(existing) = self
            .edges
            .iter_mut()
            .find(|e| e.source == source && e.target == target && e.edge_type == edge_type)
        {
            existing.weight += weight;
            if let Some(slot) = self.adjacency[si].iter_mut().find(|(t, _)| *t == ti) {
                slot.1 += weight;
            }
            if let Some(slot) = self.reverse[ti].iter_mut().find(|(s, _)| *s == si) {
                slot.1 += weight;
            }
            return;
        }

        self.edges.push(Edge {
            source: source.to_string(),
            target: target.to_string(),
            edge_type,
            weight,
        });
        self.adjacency[si].push((ti, weight));
        self.reverse[ti].push((si, weight));
    }

    pub fn node(&self, id: &str) -> Option<&Node> {
        self.index_of.get(id).map(|i| &self.nodes[*i])
    }

    pub fn contains(&self, id: &str) -> bool {
        self.index_of.contains_key(id)
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Outgoing neighbours as ids.
    pub fn successors(&self, id: &str) -> Vec<&str> {
        self.index_of
            .get(id)
            .map(|i| {
                self.adjacency[*i]
                    .iter()
                    .map(|(t, _)| self.nodes[*t].id.as_str())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Incoming neighbours as ids.
    pub fn predecessors(&self, id: &str) -> Vec<&str> {
        self.index_of
            .get(id)
            .map(|i| {
                self.reverse[*i]
                    .iter()
                    .map(|(s, _)| self.nodes[*s].id.as_str())
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn edges_by_type(&self, edge_type: EdgeType) -> usize {
        self.edges.iter().filter(|e| e.edge_type == edge_type).count()
    }

    /// Build a graph from Subsystem A's chunks.
    ///
    /// Containment comes from the parent links the chunker already resolved.
    /// Call edges come from referenced symbols, resolved by name - and an
    /// ambiguous name (same symbol defined in several places) produces **no
    /// edge**, because a wrong edge silently misdirects the random walk and is
    /// worse than a missing one.
    pub fn from_chunk_index(index: &ChunkIndex) -> Self {
        let mut graph = CodeGraph::new();

        for chunk in &index.chunks {
            graph.add_node(Node {
                id: chunk.id.clone(),
                name: chunk.name.clone(),
                type_variant: chunk.kind.as_str().to_string(),
                file_path: chunk.file_path.clone(),
                line_start: chunk.line_start,
                line_end: chunk.line_end,
            });
        }

        // Containment.
        for chunk in &index.chunks {
            if let Some(parent) = &chunk.parent {
                if graph.contains(parent) {
                    graph.add_edge(parent, &chunk.id, EdgeType::HasProperty);
                }
            }
        }

        // Name -> definitions, for call resolution.
        let mut by_name: HashMap<String, Vec<&str>> = HashMap::new();
        for chunk in &index.chunks {
            by_name
                .entry(chunk.name.to_lowercase())
                .or_default()
                .push(chunk.id.as_str());
        }

        // Calls.
        for chunk in &index.chunks {
            for symbol in &chunk.referenced_symbols {
                let candidates = match by_name.get(&symbol.to_lowercase()) {
                    Some(c) => c,
                    None => continue,
                };

                let target = if candidates.len() == 1 {
                    Some(candidates[0])
                } else {
                    // Ambiguous globally, but unambiguous within one file is a
                    // safe resolution.
                    let same_file: Vec<&&str> = candidates
                        .iter()
                        .filter(|id| {
                            index
                                .get(id)
                                .map(|c| c.file_path == chunk.file_path)
                                .unwrap_or(false)
                        })
                        .collect();
                    if same_file.len() == 1 {
                        Some(*same_file[0])
                    } else {
                        None // genuinely ambiguous: emit nothing
                    }
                };

                if let Some(target) = target {
                    if target != chunk.id {
                        graph.add_edge(&chunk.id, target, EdgeType::CallsFunction);
                    }
                }
            }
        }

        // Imports, at file granularity: a chunk referencing a symbol defined in
        // another file implies a module dependency.
        let mut file_deps: HashSet<(String, String)> = HashSet::new();
        for chunk in &index.chunks {
            for symbol in &chunk.referenced_symbols {
                if let Some(candidates) = by_name.get(&symbol.to_lowercase()) {
                    for id in candidates {
                        if let Some(target) = index.get(id) {
                            if target.file_path != chunk.file_path {
                                file_deps
                                    .insert((chunk.file_path.clone(), target.file_path.clone()));
                            }
                        }
                    }
                }
            }
        }

        // Represent a file by its outermost chunk.
        let mut file_anchor: HashMap<&str, &str> = HashMap::new();
        for chunk in &index.chunks {
            let entry = file_anchor.entry(chunk.file_path.as_str());
            entry
                .and_modify(|current| {
                    if let Some(existing) = index.get(current) {
                        if chunk.line_start < existing.line_start {
                            *current = chunk.id.as_str();
                        }
                    }
                })
                .or_insert(chunk.id.as_str());
        }

        let deps: Vec<(String, String)> = file_deps.into_iter().collect();
        for (from_file, to_file) in deps {
            if let (Some(a), Some(b)) = (
                file_anchor.get(from_file.as_str()).copied(),
                file_anchor.get(to_file.as_str()).copied(),
            ) {
                let (a, b) = (a.to_string(), b.to_string());
                graph.add_edge(&a, &b, EdgeType::ImportsModule);
            }
        }

        graph
    }
}

/// Tuning for the random walk.
#[derive(Debug, Clone)]
pub struct PprConfig {
    /// Probability of following an edge rather than teleporting to a seed.
    ///
    /// **Not 0.85.** The classic value is tuned for the web graph, where the
    /// goal is a *global* notion of importance over billions of pages. Here the
    /// graph is small, the walk is bidirectional, and the goal is the opposite:
    /// stay near the seeds. At 0.85 on a four-node call chain the mass drifts
    /// onto whichever node is most central - the middle of the chain outranks
    /// the seed the user actually asked about, which is precisely the failure
    /// personalized PageRank exists to prevent. 0.6 keeps the walk local while
    /// still propagating two to three hops, which is the useful range for
    /// "show me the code around this".
    pub damping: f64,
    /// Iteration cap. Convergence to `tolerance` needs roughly
    /// `log10(1/tolerance) / log10(1/damping)` steps - about 41 at damping 0.6
    /// and about 128 at 0.85, so the cap must comfortably exceed both or
    /// `converged` silently reports false.
    pub max_iterations: usize,
    /// L1 convergence threshold.
    pub tolerance: f64,
    /// Also traverse edges backwards, at this fraction of forward weight.
    ///
    /// Callers of a function are useful context, but weaker than callees, and
    /// reverse edges are what let mass pool on central nodes. 0.25 admits the
    /// signal without letting hubs dominate the seed.
    pub reverse_factor: f64,
}

impl Default for PprConfig {
    fn default() -> Self {
        PprConfig {
            damping: 0.6,
            max_iterations: 200,
            tolerance: 1e-9,
            reverse_factor: 0.25,
        }
    }
}

/// Outcome of a personalized PageRank run.
#[derive(Debug, Clone, Default)]
pub struct PprResult {
    /// `node_id -> score`, descending.
    pub ranked: Vec<(String, f64)>,
    pub iterations: usize,
    pub converged: bool,
    pub scores: BTreeMap<String, f64>,
}

impl PprResult {
    pub fn score(&self, id: &str) -> f64 {
        *self.scores.get(id).unwrap_or(&0.0)
    }

    pub fn top(&self, n: usize) -> &[(String, f64)] {
        &self.ranked[..n.min(self.ranked.len())]
    }
}

/// Subsystem B. Runs personalized PageRank over the semantic call graph.
#[derive(Debug, Clone, Default)]
pub struct PprEngine {
    pub config: PprConfig,
}

impl PprEngine {
    pub fn new() -> Self {
        PprEngine {
            config: PprConfig::default(),
        }
    }

    pub fn with_config(config: PprConfig) -> Self {
        PprEngine { config }
    }

    /// Power-iterate to a stationary distribution biased toward `seeds`.
    ///
    /// `seeds` maps node id to preference mass; it is normalized internally.
    /// An empty or wholly unknown seed set returns an empty result rather than
    /// silently degrading to global PageRank - answering a different question
    /// than the caller asked is worse than answering none.
    pub fn compute(&self, graph: &CodeGraph, seeds: &HashMap<String, f64>) -> PprResult {
        if graph.is_empty() {
            return PprResult::default();
        }

        // Personalization vector.
        let mut personalization = vec![0.0f64; graph.node_count()];
        let mut total = 0.0f64;
        for (id, weight) in seeds {
            if *weight <= 0.0 {
                continue;
            }
            if let Some(index) = graph.index_of.get(id) {
                personalization[*index] += *weight;
                total += *weight;
            }
        }
        if total <= 0.0 {
            return PprResult::default();
        }
        for value in personalization.iter_mut() {
            *value /= total;
        }

        // Row-normalized transition weights, forward plus dampened reverse.
        let n = graph.node_count();
        let mut transitions: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for source in 0..n {
            let mut combined: HashMap<usize, f64> = HashMap::new();
            for (target, weight) in &graph.adjacency[source] {
                *combined.entry(*target).or_insert(0.0) += *weight;
            }
            if self.config.reverse_factor > 0.0 {
                for (target, weight) in &graph.reverse[source] {
                    *combined.entry(*target).or_insert(0.0) +=
                        *weight * self.config.reverse_factor;
                }
            }
            let row_total: f64 = combined.values().sum();
            if row_total > 0.0 {
                transitions[source] = combined
                    .into_iter()
                    .map(|(target, weight)| (target, weight / row_total))
                    .collect();
                // Deterministic iteration order.
                transitions[source].sort_by_key(|(target, _)| *target);
            }
        }

        let mut rank = personalization.clone();
        let mut next = vec![0.0f64; n];
        let mut iterations = 0usize;
        let mut converged = false;

        for step in 0..self.config.max_iterations {
            iterations = step + 1;
            for value in next.iter_mut() {
                *value = 0.0;
            }

            // Mass on dangling nodes teleports home, otherwise it leaks and the
            // distribution stops summing to 1.
            let mut dangling = 0.0f64;
            for source in 0..n {
                if transitions[source].is_empty() {
                    dangling += rank[source];
                    continue;
                }
                let outgoing = rank[source];
                if outgoing == 0.0 {
                    continue;
                }
                for (target, probability) in &transitions[source] {
                    next[*target] += outgoing * probability;
                }
            }

            let d = self.config.damping;
            for i in 0..n {
                next[i] = d * (next[i] + dangling * personalization[i])
                    + (1.0 - d) * personalization[i];
            }

            let delta: f64 = next
                .iter()
                .zip(rank.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();

            std::mem::swap(&mut rank, &mut next);

            if delta < self.config.tolerance {
                converged = true;
                break;
            }
        }

        let mut scores = BTreeMap::new();
        let mut ranked: Vec<(String, f64)> = Vec::with_capacity(n);
        for (index, node) in graph.nodes.iter().enumerate() {
            if rank[index] > 0.0 {
                scores.insert(node.id.clone(), rank[index]);
                ranked.push((node.id.clone(), rank[index]));
            }
        }
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        PprResult {
            ranked,
            iterations,
            converged,
            scores,
        }
    }

    /// Convenience: seed from `(id, score)` pairs, e.g. vector search output.
    pub fn run_from_matches(&self, graph: &CodeGraph, matches: &[(String, f32)]) -> PprResult {
        let seeds: HashMap<String, f64> = matches
            .iter()
            .filter(|(_, score)| *score > 0.0)
            .map(|(id, score)| (id.clone(), *score as f64))
            .collect();
        self.compute(graph, &seeds)
    }
}

/// Input bundle for the [`Filter`] implementation.
///
/// The graph is held behind an [`Arc`] rather than a reference: an associated
/// type cannot carry a lifetime that the impl does not constrain, and cloning
/// an `Arc` is free.
#[derive(Debug, Clone)]
pub struct PprInput {
    pub graph: Arc<CodeGraph>,
    pub seeds: HashMap<String, f64>,
}

impl PprInput {
    pub fn new(graph: Arc<CodeGraph>, seeds: HashMap<String, f64>) -> Self {
        PprInput { graph, seeds }
    }
}

impl Filter for PprEngine {
    type Input = PprInput;
    type Output = PprResult;

    fn name(&self) -> &'static str {
        "ppr_graph"
    }

    fn apply(
        &self,
        input: PprInput,
        ctx: &mut PipelineContext,
    ) -> Result<PprResult, PipelineError> {
        if input.graph.is_empty() {
            return Err(PipelineError::Empty("ppr_graph"));
        }
        let result = self.compute(&input.graph, &input.seeds);
        ctx.set("ppr_iterations", result.iterations as u64);
        ctx.set("ppr_scored_nodes", result.ranked.len() as u64);
        if !result.converged && !result.ranked.is_empty() {
            ctx.note(format!(
                "PPR hit the {}-iteration cap without converging",
                self.config.max_iterations
            ));
        }
        Ok(result)
    }
}

/// Helper so callers can classify a chunk kind into an edge relationship.
pub fn containment_edge_for(kind: ScopeKind) -> EdgeType {
    if kind.is_callable() {
        EdgeType::CallsFunction
    } else {
        EdgeType::HasProperty
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::ast_chunker::{AstChunker, SourceFile};

    fn node(id: &str) -> Node {
        Node {
            id: id.to_string(),
            name: id.to_string(),
            type_variant: "function".to_string(),
            file_path: "m.rs".to_string(),
            line_start: 1,
            line_end: 5,
        }
    }

    fn chain_graph() -> CodeGraph {
        let mut g = CodeGraph::new();
        for id in ["a", "b", "c", "d", "island"] {
            g.add_node(node(id));
        }
        g.add_edge("a", "b", EdgeType::CallsFunction);
        g.add_edge("b", "c", EdgeType::CallsFunction);
        g.add_edge("c", "d", EdgeType::CallsFunction);
        g
    }

    #[test]
    fn edge_weights_match_the_specification() {
        assert_eq!(EdgeType::CallsFunction.weight(), 3.0);
        assert_eq!(EdgeType::ImportsModule.weight(), 2.0);
        assert_eq!(EdgeType::HasProperty.weight(), 1.0);
    }

    #[test]
    fn duplicate_nodes_are_deduplicated() {
        let mut g = CodeGraph::new();
        g.add_node(node("a"));
        g.add_node(node("a"));
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn repeated_edges_accumulate_weight() {
        let mut g = CodeGraph::new();
        g.add_node(node("a"));
        g.add_node(node("b"));
        g.add_edge("a", "b", EdgeType::CallsFunction);
        g.add_edge("a", "b", EdgeType::CallsFunction);
        assert_eq!(g.edge_count(), 1);
        assert_eq!(g.edges[0].weight, 6.0);
    }

    #[test]
    fn self_loops_are_rejected() {
        let mut g = CodeGraph::new();
        g.add_node(node("a"));
        g.add_edge("a", "a", EdgeType::CallsFunction);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn edges_to_unknown_nodes_are_dropped() {
        let mut g = CodeGraph::new();
        g.add_node(node("a"));
        g.add_edge("a", "ghost", EdgeType::CallsFunction);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn successors_and_predecessors_are_inverses() {
        let g = chain_graph();
        assert_eq!(g.successors("a"), vec!["b"]);
        assert_eq!(g.predecessors("b"), vec!["a"]);
    }

    #[test]
    fn scores_form_a_probability_distribution() {
        let g = chain_graph();
        let seeds: HashMap<String, f64> = [("a".to_string(), 1.0)].into_iter().collect();
        let result = PprEngine::new().compute(&g, &seeds);
        let total: f64 = result.scores.values().sum();
        assert!((total - 1.0).abs() < 1e-6, "mass was {}", total);
        assert!(result.scores.values().all(|v| *v >= 0.0));
    }

    #[test]
    fn seed_receives_the_highest_score() {
        let g = chain_graph();
        let seeds: HashMap<String, f64> = [("a".to_string(), 1.0)].into_iter().collect();
        let result = PprEngine::new().compute(&g, &seeds);
        assert_eq!(result.ranked[0].0, "a");
    }

    #[test]
    fn score_decays_along_the_call_chain() {
        let g = chain_graph();
        let seeds: HashMap<String, f64> = [("a".to_string(), 1.0)].into_iter().collect();
        let r = PprEngine::new().compute(&g, &seeds);
        assert!(r.score("b") > r.score("c"), "b={} c={}", r.score("b"), r.score("c"));
        assert!(r.score("c") > r.score("d"));
    }

    #[test]
    fn disconnected_nodes_get_no_mass() {
        let g = chain_graph();
        let seeds: HashMap<String, f64> = [("a".to_string(), 1.0)].into_iter().collect();
        let result = PprEngine::new().compute(&g, &seeds);
        assert!(result.score("island") < 1e-9, "island got {}", result.score("island"));
    }

    #[test]
    fn personalization_actually_personalizes() {
        // If this failed, we would be computing global PageRank.
        let g = chain_graph();
        let from_a = PprEngine::new().compute(&g, &[("a".to_string(), 1.0)].into_iter().collect());
        let from_d = PprEngine::new().compute(&g, &[("d".to_string(), 1.0)].into_iter().collect());
        assert_ne!(from_a.ranked[0].0, from_d.ranked[0].0);
    }

    #[test]
    fn empty_seeds_return_nothing() {
        let g = chain_graph();
        let result = PprEngine::new().compute(&g, &HashMap::new());
        assert!(result.ranked.is_empty());
    }

    #[test]
    fn unknown_seeds_return_nothing() {
        let g = chain_graph();
        let seeds: HashMap<String, f64> = [("ghost".to_string(), 1.0)].into_iter().collect();
        assert!(PprEngine::new().compute(&g, &seeds).ranked.is_empty());
    }

    #[test]
    fn converges_well_before_the_cap() {
        let g = chain_graph();
        let seeds: HashMap<String, f64> = [("a".to_string(), 1.0)].into_iter().collect();
        let result = PprEngine::new().compute(&g, &seeds);
        assert!(result.converged, "did not converge in {} iterations", result.iterations);
        // ~42 at damping 0.6; the cap is 200.
        assert!(result.iterations < 100, "took {} iterations", result.iterations);
    }

    #[test]
    fn seed_outranks_central_nodes() {
        // Regression guard. At damping 0.85 with reverse traversal the middle
        // of the chain outranks the seed, which silently turns personalized
        // PageRank back into global PageRank.
        let g = chain_graph();
        let seeds: HashMap<String, f64> = [("a".to_string(), 1.0)].into_iter().collect();
        let r = PprEngine::new().compute(&g, &seeds);
        assert!(
            r.score("a") > r.score("c"),
            "central node beat the seed: a={} c={}",
            r.score("a"),
            r.score("c")
        );
    }

    #[test]
    fn lower_damping_keeps_more_mass_on_the_seed() {
        let g = chain_graph();
        let seeds: HashMap<String, f64> = [("a".to_string(), 1.0)].into_iter().collect();
        let low = PprEngine::with_config(PprConfig { damping: 0.1, ..Default::default() })
            .compute(&g, &seeds);
        let high = PprEngine::with_config(PprConfig { damping: 0.95, ..Default::default() })
            .compute(&g, &seeds);
        assert!(low.score("a") > high.score("a"));
    }

    #[test]
    fn heavier_edges_carry_more_mass() {
        // Same topology, different relationship: CALLS must beat HAS_PROPERTY.
        let mut calls = CodeGraph::new();
        calls.add_node(node("seed"));
        calls.add_node(node("x"));
        calls.add_node(node("y"));
        calls.add_edge("seed", "x", EdgeType::CallsFunction);
        calls.add_edge("seed", "y", EdgeType::HasProperty);

        let seeds: HashMap<String, f64> = [("seed".to_string(), 1.0)].into_iter().collect();
        let result = PprEngine::new().compute(&calls, &seeds);
        assert!(
            result.score("x") > result.score("y"),
            "calls={} property={}",
            result.score("x"),
            result.score("y")
        );
    }

    #[test]
    fn dangling_mass_is_conserved() {
        let mut g = CodeGraph::new();
        g.add_node(node("a"));
        g.add_node(node("sink"));
        g.add_edge("a", "sink", EdgeType::CallsFunction);
        let engine = PprEngine::with_config(PprConfig { reverse_factor: 0.0, ..Default::default() });
        let seeds: HashMap<String, f64> = [("a".to_string(), 1.0)].into_iter().collect();
        let total: f64 = engine.compute(&g, &seeds).scores.values().sum();
        assert!((total - 1.0).abs() < 1e-6, "mass leaked: {}", total);
    }

    #[test]
    fn multi_seed_blends_both_neighbourhoods() {
        let g = chain_graph();
        let seeds: HashMap<String, f64> =
            [("a".to_string(), 1.0), ("d".to_string(), 1.0)].into_iter().collect();
        let result = PprEngine::new().compute(&g, &seeds);
        assert!(result.score("a") > 0.0 && result.score("d") > 0.0);
    }

    #[test]
    fn empty_graph_is_safe() {
        let g = CodeGraph::new();
        let seeds: HashMap<String, f64> = [("a".to_string(), 1.0)].into_iter().collect();
        assert!(PprEngine::new().compute(&g, &seeds).ranked.is_empty());
    }

    #[test]
    fn builds_a_graph_from_real_chunks() {
        let src = "fn helper() -> u32 { 7 }\n\nfn caller() -> u32 { helper() }\n";
        let chunks = AstChunker::new()
            .chunk_file(&SourceFile::new("m.rs", src))
            .unwrap();
        let graph = CodeGraph::from_chunk_index(&ChunkIndex::new(chunks));
        assert_eq!(graph.node_count(), 2);
        assert!(
            graph.successors("m.rs::caller").contains(&"m.rs::helper"),
            "edges: {:?}",
            graph.edges
        );
    }

    #[test]
    fn containment_produces_has_property_edges() {
        let src = "struct S { a: u32 }\nimpl S {\n    fn go(&self) {}\n}\n";
        let chunks = AstChunker::new()
            .chunk_file(&SourceFile::new("s.rs", src))
            .unwrap();
        let graph = CodeGraph::from_chunk_index(&ChunkIndex::new(chunks));
        assert!(graph.edges_by_type(EdgeType::HasProperty) > 0);
    }

    #[test]
    fn ambiguous_call_names_produce_no_edge() {
        let src = "mod a {\n    pub fn run() {}\n}\nmod b {\n    pub fn run() {}\n}\nfn go() { run(); }\n";
        let chunks = AstChunker::new()
            .chunk_file(&SourceFile::new("m.rs", src))
            .unwrap();
        let graph = CodeGraph::from_chunk_index(&ChunkIndex::new(chunks));
        let call_targets = graph.successors("m.rs::go");
        assert!(
            !call_targets.iter().any(|t| t.ends_with("run")),
            "ambiguous name was guessed: {:?}",
            call_targets
        );
    }

    #[test]
    fn filter_records_telemetry() {
        let g = chain_graph();
        let mut ctx = PipelineContext::new();
        let input = PprInput::new(
            Arc::new(g),
            [("a".to_string(), 1.0)].into_iter().collect(),
        );
        let result = PprEngine::new().run(input, &mut ctx).unwrap();
        assert!(!result.ranked.is_empty());
        assert!(ctx.counter("ppr_iterations") > 0);
        assert!(ctx.stage_micros.contains_key("ppr_graph"));
    }

    #[test]
    fn run_from_matches_accepts_vector_output() {
        let g = chain_graph();
        let matches = vec![("a".to_string(), 0.9f32), ("ghost".to_string(), 0.4f32)];
        let result = PprEngine::new().run_from_matches(&g, &matches);
        assert_eq!(result.ranked[0].0, "a");
    }
}
