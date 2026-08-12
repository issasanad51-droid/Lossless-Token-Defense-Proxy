//! Search subsystem: AST scope mapping, PPR call-chain walk, RRF fusion.
//!
//! [`RetrievalEngine`] is the integrated code-graph façade. It owns the
//! parser, the graph, the vectorizer and the blender, and exposes a single
//! `index_file` / `finalize` / `query` lifecycle so the orchestrator never
//! has to wire the three rankers by hand.

pub mod ast_parser;
pub mod ppr_graph;
pub mod rrf_blender;

#[allow(unused_imports)]
pub use ast_parser::{AstParser, ScopeChunk, ScopeKind, SourceLang};
#[allow(unused_imports)]
pub use ppr_graph::{
    EdgeKind, GraphEdge, GraphNode, NodeId, PprGraph, SemanticVectorizer, CALLS_FUNCTION,
    HAS_PROPERTY, IMPORTS_MODULE,
};
#[allow(unused_imports)]
pub use rrf_blender::{
    BlendedCandidate, RankedItem, RankedList, RrfBlender, CONFIDENCE_OVERRIDE, DEFAULT_K,
    PROXIMITY_WINDOW,
};

use regex::Regex;
use std::collections::HashMap;

/// End-to-end retrieval façade.
pub struct RetrievalEngine {
    pub parser: AstParser,
    pub graph: PprGraph,
    pub blender: RrfBlender,
    pub vectorizer: SemanticVectorizer,
    pub chunks: Vec<ScopeChunk>,
    pub sources: HashMap<String, String>,
    finalized: bool,
}

impl std::fmt::Debug for RetrievalEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetrievalEngine")
            .field("files", &self.sources.len())
            .field("chunks", &self.chunks.len())
            .field("nodes", &self.graph.node_count())
            .field("edges", &self.graph.edge_count())
            .field("finalized", &self.finalized)
            .finish()
    }
}

impl RetrievalEngine {
    pub fn new() -> Self {
        Self {
            parser: AstParser::new(),
            graph: PprGraph::new(),
            blender: RrfBlender::new(),
            vectorizer: SemanticVectorizer::new(),
            chunks: Vec::new(),
            sources: HashMap::new(),
            finalized: false,
        }
    }

    /// Parse `source` into immutable scope chunks and park it for wiring.
    pub fn index_file(&mut self, path: &str, source: &str) {
        let chunks = self.parser.parse_file(path, source);
        self.chunks.extend(chunks);
        self.sources.insert(path.to_string(), source.to_string());
        self.finalized = false;
    }

    /// Fit the vectorizer, ingest nodes, and wire language-execution edges.
    pub fn finalize(&mut self) {
        let docs: Vec<&str> = self
            .chunks
            .iter()
            .map(|c| c.source.as_str())
            .collect();
        self.vectorizer.fit(&docs);
        self.graph = PprGraph::new();
        self.graph.ingest_chunks(&self.chunks);
        self.graph
            .wire_language_edges(&self.chunks, &self.sources);
        self.finalized = true;
    }

    /// Run vector + AST + regex rankers, bias PPR at the vector hits, and
    /// fuse everything with overlap-reconciling RRF.
    pub fn query(&mut self, query: &str, top_k: usize) -> QueryOutcome {
        if !self.finalized {
            self.finalize();
        }

        let vector = self.rank_vector(query);
        let ast = self.rank_ast(query);
        let regex = self.rank_regex(query);

        // Personalization focus: every vector hit above a modest floor.
        let mut hits = Vec::new();
        for item in &vector.items {
            if let Some(id) = self.graph.id_of(&item.id) {
                hits.push((id, item.confidence));
            }
        }
        let focus = self.graph.focus_from_hits(&hits, 0.05);
        let ppr_ranked = self
            .graph
            .personalized_pagerank(&focus, 0.85, 50, 1e-12);

        // Promote PPR-surfaced structural neighbours into the vector list
        // so the blender sees them without inventing a fourth ranker.
        let mut vector = vector;
        let already: HashMap<String, ()> =
            vector.items.iter().map(|i| (i.id.clone(), ())).collect();
        let mut extra_rank = vector.items.len() + 1;
        for (node_id, mass) in ppr_ranked.iter().take(top_k.saturating_mul(3).max(8)) {
            let node = &self.graph.nodes[*node_id];
            if already.contains_key(&node.key) {
                continue;
            }
            if let Some(chunk) = self.chunks.iter().find(|c| c.id() == node.key) {
                vector.items.push(RankedItem {
                    id: chunk.id(),
                    file_path: chunk.file_path.clone(),
                    line_start: chunk.line_start,
                    line_end: chunk.line_end,
                    content: chunk.source.clone(),
                    rank: extra_rank,
                    confidence: (*mass).min(0.94), // never auto-override via PPR mass
                    source_ranker: "ppr_expansion".into(),
                });
                extra_rank += 1;
            }
        }

        let blended = self
            .blender
            .blend(&[vector.clone(), ast.clone(), regex.clone()], &self.sources);

        QueryOutcome {
            query: query.to_string(),
            vector,
            ast,
            regex,
            ppr: ppr_ranked,
            blended: blended.into_iter().take(top_k).collect(),
            graph_nodes: self.graph.node_count(),
            graph_edges: self.graph.edge_count(),
        }
    }

    fn rank_vector(&self, query: &str) -> RankedList {
        let mut scored: Vec<(f64, &ScopeChunk)> = self
            .chunks
            .iter()
            .map(|c| {
                let blob = format!("{} {} {}", c.name, c.kind.as_str(), c.source);
                (self.vectorizer.cosine(query, &blob), c)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let items = scored
            .into_iter()
            .enumerate()
            .map(|(i, (sim, c))| RankedItem {
                id: c.id(),
                file_path: c.file_path.clone(),
                line_start: c.line_start,
                line_end: c.line_end,
                content: c.source.clone(),
                rank: i + 1,
                confidence: sim,
                source_ranker: "vector".into(),
            })
            .collect();
        RankedList::new("vector", 1.15, items)
    }

    fn rank_ast(&self, query: &str) -> RankedList {
        let tokens = tokenize_query(query);
        let mut scored: Vec<(f64, &ScopeChunk)> = Vec::new();
        for chunk in &self.chunks {
            let name = chunk.name.to_ascii_lowercase();
            let mut best: f64 = 0.0;
            for tok in &tokens {
                if name == *tok {
                    best = best.max(1.0);
                } else if name.starts_with(tok.as_str()) || tok.starts_with(&name) {
                    best = best.max(0.88);
                } else if name.contains(tok.as_str()) {
                    best = best.max(0.72);
                } else {
                    let sim = levenshtein_sim(&name, tok);
                    if sim >= 0.6 {
                        best = best.max(sim * 0.85);
                    }
                }
            }
            if best > 0.0 {
                scored.push((best, chunk));
            }
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let items = scored
            .into_iter()
            .enumerate()
            .map(|(i, (sim, c))| RankedItem {
                id: c.id(),
                file_path: c.file_path.clone(),
                line_start: c.line_start,
                line_end: c.line_end,
                content: c.source.clone(),
                rank: i + 1,
                confidence: sim,
                source_ranker: "ast".into(),
            })
            .collect();
        RankedList::new("ast", 1.00, items)
    }

    fn rank_regex(&self, query: &str) -> RankedList {
        let tokens = tokenize_query(query);
        if tokens.is_empty() {
            return RankedList::new("regex", 0.85, Vec::new());
        }
        let pat = tokens
            .iter()
            .map(|t| regex::escape(t))
            .collect::<Vec<_>>()
            .join("|");
        let re = match Regex::new(&format!(r"(?i)\b(?:{pat})\b")) {
            Ok(r) => r,
            Err(_) => return RankedList::new("regex", 0.85, Vec::new()),
        };
        let mut scored: Vec<(f64, usize, &ScopeChunk)> = Vec::new();
        for chunk in &self.chunks {
            let hits = re.find_iter(&chunk.source).count();
            if hits == 0 {
                continue;
            }
            let density = hits as f64 / (chunk.source.split_whitespace().count().max(1) as f64);
            let conf = (hits as f64 / 4.0).min(1.0).max(density.min(1.0));
            scored.push((conf, hits, chunk));
        }
        scored.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then_with(|| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal))
        });
        let items = scored
            .into_iter()
            .enumerate()
            .map(|(i, (conf, _, c))| RankedItem {
                id: c.id(),
                file_path: c.file_path.clone(),
                line_start: c.line_start,
                line_end: c.line_end,
                content: c.source.clone(),
                rank: i + 1,
                confidence: conf,
                source_ranker: "regex".into(),
            })
            .collect();
        RankedList::new("regex", 0.85, items)
    }
}

impl Default for RetrievalEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Bundle returned by [`RetrievalEngine::query`].
#[derive(Debug, Clone)]
pub struct QueryOutcome {
    pub query: String,
    pub vector: RankedList,
    pub ast: RankedList,
    pub regex: RankedList,
    pub ppr: Vec<(NodeId, f64)>,
    pub blended: Vec<BlendedCandidate>,
    pub graph_nodes: usize,
    pub graph_edges: usize,
}

impl QueryOutcome {
    /// Concatenate the blended windows — this is the optimized context
    /// the proxy would forward to a model.
    pub fn context(&self) -> String {
        let mut out = String::new();
        for (i, c) in self.blended.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            out.push_str(&format!(
                "# {} {}:{}-{}\n{}",
                c.file_path, c.id, c.line_start, c.line_end, c.content
            ));
            out.push('\n');
        }
        out
    }
}

fn tokenize_query(q: &str) -> Vec<String> {
    q.split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|t| t.len() >= 2)
        .map(|t| t.to_ascii_lowercase())
        .collect()
}

fn levenshtein_sim(a: &str, b: &str) -> f64 {
    let aa: Vec<char> = a.chars().collect();
    let bb: Vec<char> = b.chars().collect();
    if aa.is_empty() && bb.is_empty() {
        return 1.0;
    }
    let m = aa.len();
    let n = bb.len();
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if aa[i - 1] == bb[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    let dist = prev[n] as f64;
    let max = m.max(n) as f64;
    if max == 0.0 {
        1.0
    } else {
        (1.0 - dist / max).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_returns_auth_related_chunks() {
        let rust = r#"
pub fn authenticate(token: &str) -> bool {
    validate_token(token)
}
pub fn validate_token(token: &str) -> bool {
    decode_jwt(token).is_some()
}
pub fn decode_jwt(token: &str) -> Option<u64> { Some(1) }
pub fn render_template(name: &str) -> String { name.to_string() }
"#;
        let mut eng = RetrievalEngine::new();
        eng.index_file("auth.rs", rust);
        eng.finalize();
        let out = eng.query("authenticate user token jwt", 4);
        assert!(!out.blended.is_empty());
        let blob = out
            .blended
            .iter()
            .map(|c| c.content.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            blob.contains("authenticate") || blob.contains("validate_token") || blob.contains("decode_jwt"),
            "retrieval missed the auth chain:\n{blob}"
        );
    }
}
