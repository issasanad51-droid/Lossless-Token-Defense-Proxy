//! Semantic-weighted Personalized PageRank over a call-chain graph.
//!
//! The graph is built from scratch: nodes are methods / modules / types
//! and edges carry hardcoded directional language-execution weights
//!
//! * [`CALLS_FUNCTION`]  — 3.0
//! * [`IMPORTS_MODULE`]  — 2.0
//! * [`HAS_PROPERTY`]    — 1.0
//!
//! Personalized PageRank then walks the graph with a focus vector taken
//! directly from vector-semantic similarity hits, so a single power
//! iteration surfaces the structural neighbourhood of the query.

use std::collections::HashMap;

use crate::indexer::ast_parser::{
    contains_call, extract_imports, ScopeChunk, ScopeKind,
};

/// Directional execution weight: caller → callee.
pub const CALLS_FUNCTION: f64 = 3.0;
/// Directional execution weight: file / module → imported symbol.
pub const IMPORTS_MODULE: f64 = 2.0;
/// Directional execution weight: type / impl → field or method.
pub const HAS_PROPERTY: f64 = 1.0;

/// Hardcoded language-execution edge taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    CallsFunction,
    ImportsModule,
    HasProperty,
}

impl EdgeKind {
    pub fn weight(self) -> f64 {
        match self {
            EdgeKind::CallsFunction => CALLS_FUNCTION,
            EdgeKind::ImportsModule => IMPORTS_MODULE,
            EdgeKind::HasProperty => HAS_PROPERTY,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            EdgeKind::CallsFunction => "CALLS_FUNCTION",
            EdgeKind::ImportsModule => "IMPORTS_MODULE",
            EdgeKind::HasProperty => "HAS_PROPERTY",
        }
    }
}

/// Stable numeric identifier of a graph node.
pub type NodeId = usize;

/// A method, type, or module vertex.
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: NodeId,
    pub key: String,
    pub name: String,
    pub kind: ScopeKind,
    pub file_path: String,
    pub line_start: usize,
    pub line_end: usize,
}

/// A directed, weighted dependency.
#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub from: NodeId,
    pub to: NodeId,
    pub kind: EdgeKind,
    pub weight: f64,
}

/// Directed call-chain graph plus the PPR walker.
#[derive(Debug, Clone, Default)]
pub struct PprGraph {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    key_to_id: HashMap<String, NodeId>,
    name_index: HashMap<String, Vec<NodeId>>,
    /// Adjacency: from → (to, weight) aggregated over parallel edges.
    adj: Vec<Vec<(NodeId, f64)>>,
    out_sum: Vec<f64>,
}

impl PprGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn id_of(&self, key: &str) -> Option<NodeId> {
        self.key_to_id.get(key).copied()
    }

    pub fn nodes_named(&self, name: &str) -> &[NodeId] {
        self.name_index
            .get(name)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Insert a node for every scope chunk. A synthetic module node is also
    /// created per file so `IMPORTS_MODULE` edges have a well-defined source.
    pub fn ingest_chunks(&mut self, chunks: &[ScopeChunk]) {
        let mut files = Vec::new();
        for chunk in chunks {
            if !files.iter().any(|f: &String| f == &chunk.file_path) {
                files.push(chunk.file_path.clone());
            }
            self.upsert_node(
                chunk.id(),
                chunk.name.clone(),
                chunk.kind,
                chunk.file_path.clone(),
                chunk.line_start,
                chunk.line_end,
            );
        }
        for file in files {
            let key = format!("module::{file}");
            let stem = file
                .rsplit(['/', '\\'])
                .next()
                .unwrap_or(&file)
                .to_string();
            self.upsert_node(key, stem, ScopeKind::Module, file, 1, 1);
        }
        self.rebuild_adj();
    }

    fn upsert_node(
        &mut self,
        key: String,
        name: String,
        kind: ScopeKind,
        file_path: String,
        line_start: usize,
        line_end: usize,
    ) -> NodeId {
        if let Some(&id) = self.key_to_id.get(&key) {
            return id;
        }
        let id = self.nodes.len();
        self.key_to_id.insert(key.clone(), id);
        self.name_index
            .entry(name.clone())
            .or_default()
            .push(id);
        self.nodes.push(GraphNode {
            id,
            key,
            name,
            kind,
            file_path,
            line_start,
            line_end,
        });
        id
    }

    /// Wire CALLS_FUNCTION / IMPORTS_MODULE / HAS_PROPERTY edges from the
    /// already-ingested chunks and their source text.
    pub fn wire_language_edges(&mut self, chunks: &[ScopeChunk], sources: &HashMap<String, String>) {
        // CALLS_FUNCTION: body of A mentions B( ...
        for caller in chunks {
            let Some(&from) = self.key_to_id.get(&caller.id()) else {
                continue;
            };
            for callee in chunks {
                if caller.id() == callee.id() {
                    continue;
                }
                if !callee.kind.is_callable() && callee.kind != ScopeKind::Struct {
                    // Allow calls into constructors / types too, but only
                    // when the name is used as a call.
                    if !matches!(
                        callee.kind,
                        ScopeKind::Class | ScopeKind::Enum | ScopeKind::Impl
                    ) {
                        continue;
                    }
                }
                if contains_call(&caller.source, &callee.name) {
                    self.push_edge(from, self.key_to_id[&callee.id()], EdgeKind::CallsFunction);
                }
            }
        }

        // IMPORTS_MODULE: file module → imported names that exist as nodes.
        for (file, source) in sources {
            let module_key = format!("module::{file}");
            let Some(&from) = self.key_to_id.get(&module_key) else {
                continue;
            };
            for imported in extract_imports(source) {
                let targets: Vec<NodeId> = self.nodes_named(&imported).to_vec();
                for to in targets {
                    if to != from {
                        self.push_edge(from, to, EdgeKind::ImportsModule);
                    }
                }
            }
        }

        // HAS_PROPERTY: a type / impl / class owns every callable whose
        // line range sits strictly inside its own range in the same file.
        for owner in chunks {
            if !matches!(
                owner.kind,
                ScopeKind::Struct | ScopeKind::Enum | ScopeKind::Class | ScopeKind::Impl | ScopeKind::Trait
            ) {
                continue;
            }
            let Some(&from) = self.key_to_id.get(&owner.id()) else {
                continue;
            };
            for child in chunks {
                if child.file_path != owner.file_path || child.id() == owner.id() {
                    continue;
                }
                let inside = child.line_start >= owner.line_start
                    && child.line_end <= owner.line_end
                    && (child.line_start > owner.line_start || child.line_end < owner.line_end);
                if inside {
                    self.push_edge(from, self.key_to_id[&child.id()], EdgeKind::HasProperty);
                }
            }
            // Struct field identifiers: `name:` or `name: Type` inside the body.
            if owner.kind == ScopeKind::Struct {
                for field in struct_fields(&owner.source) {
                    // Property nodes are lightweight and keyed per owner.
                    let key = format!("{}#field#{field}", owner.id());
                    let to = self.upsert_node(
                        key,
                        field,
                        ScopeKind::Struct,
                        owner.file_path.clone(),
                        owner.line_start,
                        owner.line_end,
                    );
                    self.push_edge(from, to, EdgeKind::HasProperty);
                }
            }
        }

        self.rebuild_adj();
    }

    fn push_edge(&mut self, from: NodeId, to: NodeId, kind: EdgeKind) {
        if from == to {
            return;
        }
        // Dedup identical (from, to, kind) triples.
        if self
            .edges
            .iter()
            .any(|e| e.from == from && e.to == to && e.kind == kind)
        {
            return;
        }
        self.edges.push(GraphEdge {
            from,
            to,
            kind,
            weight: kind.weight(),
        });
    }

    fn rebuild_adj(&mut self) {
        let n = self.nodes.len();
        self.adj = vec![Vec::new(); n];
        self.out_sum = vec![0.0; n];
        for edge in &self.edges {
            if edge.from >= n || edge.to >= n {
                continue;
            }
            // Aggregate parallel edges of different kinds.
            if let Some(slot) = self.adj[edge.from].iter_mut().find(|(t, _)| *t == edge.to) {
                slot.1 += edge.weight;
            } else {
                self.adj[edge.from].push((edge.to, edge.weight));
            }
            self.out_sum[edge.from] += edge.weight;
        }
    }

    /// Personalized PageRank power iteration.
    ///
    /// `focus` is a sparse personalization vector (node → raw mass). It is
    /// L1-normalized internally. Dangling nodes redistribute their mass
    /// onto the personalization vector so the walk stays on the query's
    /// structural neighbourhood.
    pub fn personalized_pagerank(
        &self,
        focus: &HashMap<NodeId, f64>,
        damping: f64,
        iterations: usize,
        tol: f64,
    ) -> Vec<(NodeId, f64)> {
        let n = self.nodes.len();
        if n == 0 {
            return Vec::new();
        }
        let damping = damping.clamp(0.0, 1.0);

        let mut personal = vec![0.0; n];
        let mut mass = 0.0;
        for (&id, &w) in focus {
            if id < n && w > 0.0 {
                personal[id] += w;
                mass += w;
            }
        }
        if mass <= 0.0 {
            let u = 1.0 / n as f64;
            personal.fill(u);
        } else {
            for p in personal.iter_mut() {
                *p /= mass;
            }
        }

        let mut rank = personal.clone();
        let mut next = vec![0.0; n];

        for _ in 0..iterations {
            next.fill(0.0);
            let mut dangling = 0.0;
            for u in 0..n {
                if self.out_sum[u] <= 0.0 {
                    dangling += rank[u];
                    continue;
                }
                let scale = rank[u] / self.out_sum[u];
                for &(v, w) in &self.adj[u] {
                    next[v] += scale * w;
                }
            }
            let mut delta = 0.0;
            for v in 0..n {
                let val = (1.0 - damping) * personal[v] + damping * (next[v] + dangling * personal[v]);
                delta += (val - rank[v]).abs();
                rank[v] = val;
            }
            if delta < tol {
                break;
            }
        }

        let mut scored: Vec<(NodeId, f64)> = rank.into_iter().enumerate().collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// Build a focus vector from (node_id, similarity) hits. Values below
    /// `min_sim` are dropped so noise does not flatten the walk.
    pub fn focus_from_hits(&self, hits: &[(NodeId, f64)], min_sim: f64) -> HashMap<NodeId, f64> {
        let mut focus = HashMap::new();
        for &(id, sim) in hits {
            if id < self.nodes.len() && sim >= min_sim {
                *focus.entry(id).or_insert(0.0) += sim;
            }
        }
        focus
    }
}

fn struct_fields(source: &str) -> Vec<String> {
    // Very small field extractor: `name: Type` at struct-body indent.
    let mut fields = Vec::new();
    let mut in_body = false;
    for line in source.lines() {
        let t = line.trim();
        if t.contains('{') {
            in_body = true;
            continue;
        }
        if t.starts_with('}') {
            break;
        }
        if !in_body {
            continue;
        }
        let t = t.trim_start_matches("pub ").trim();
        if let Some(colon) = t.find(':') {
            let name = t[..colon].trim();
            if !name.is_empty()
                && name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                fields.push(name.to_string());
            }
        }
    }
    fields
}

/// Character n-gram + token unigram hasher used to derive the semantic
/// similarity hits that seed PPR.
#[derive(Debug, Clone, Default)]
pub struct SemanticVectorizer {
    idf: HashMap<u64, f64>,
    n_docs: usize,
}

impl SemanticVectorizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fit(&mut self, docs: &[&str]) {
        self.n_docs = docs.len();
        let mut df: HashMap<u64, usize> = HashMap::new();
        for doc in docs {
            let mut seen = HashMap::new();
            for (h, _) in features(doc) {
                seen.insert(h, ());
            }
            for h in seen.into_keys() {
                *df.entry(h).or_insert(0) += 1;
            }
        }
        self.idf.clear();
        let n = (self.n_docs as f64).max(1.0);
        for (h, d) in df {
            self.idf.insert(h, ((n + 1.0) / (d as f64 + 1.0)).ln() + 1.0);
        }
    }

    pub fn embed(&self, text: &str) -> HashMap<u64, f64> {
        let mut vec = HashMap::new();
        for (h, tf) in features(text) {
            let idf = self.idf.get(&h).copied().unwrap_or(1.0);
            *vec.entry(h).or_insert(0.0) += tf * idf;
        }
        vec
    }

    pub fn cosine(&self, a: &str, b: &str) -> f64 {
        cosine_sparse(&self.embed(a), &self.embed(b))
    }
}

fn features(text: &str) -> Vec<(u64, f64)> {
    let lower = text.to_ascii_lowercase();
    let mut acc: HashMap<u64, f64> = HashMap::new();
    // Token unigrams + bigrams.
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|t| t.len() >= 2)
        .collect();
    for t in &tokens {
        *acc.entry(fnv1a(t.as_bytes())).or_insert(0.0) += 1.0;
    }
    for pair in tokens.windows(2) {
        let mut buf = String::with_capacity(pair[0].len() + pair[1].len() + 1);
        buf.push_str(pair[0]);
        buf.push(' ');
        buf.push_str(pair[1]);
        *acc.entry(fnv1a(buf.as_bytes())).or_insert(0.0) += 1.0;
    }
    // Character trigrams on the raw lowercased text (no spaces stripped so
    // punctuation still contributes a little shape signal).
    let chars: Vec<char> = lower.chars().filter(|c| !c.is_control()).collect();
    if chars.len() >= 3 {
        for w in chars.windows(3) {
            let mut buf = [0u8; 12];
            let mut n = 0;
            for ch in w {
                let s = ch.encode_utf8(&mut buf[n..]);
                n += s.len();
            }
            *acc.entry(fnv1a(&buf[..n])).or_insert(0.0) += 1.0;
        }
    }
    acc.into_iter().collect()
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn cosine_sparse(a: &HashMap<u64, f64>, b: &HashMap<u64, f64>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let (small, large) = if a.len() < b.len() { (a, b) } else { (b, a) };
    let mut dot = 0.0;
    for (k, va) in small {
        if let Some(vb) = large.get(k) {
            dot += va * vb;
        }
    }
    let na = a.values().map(|v| v * v).sum::<f64>().sqrt();
    let nb = b.values().map(|v| v * v).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        (dot / (na * nb)).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::ast_parser::AstParser;

    #[test]
    fn ppr_biases_toward_callees_of_focus() {
        let src = r#"
fn authenticate(tok: &str) -> bool {
    validate_token(tok) && lookup_user(tok)
}
fn validate_token(tok: &str) -> bool { decode_jwt(tok).is_some() }
fn decode_jwt(tok: &str) -> Option<u64> { Some(tok.len() as u64) }
fn lookup_user(tok: &str) -> bool { !tok.is_empty() }
fn unrelated() { let _x = 1; }
"#;
        let parser = AstParser::new();
        let chunks = parser.parse_file("g.rs", src);
        let mut graph = PprGraph::new();
        graph.ingest_chunks(&chunks);
        let mut sources = HashMap::new();
        sources.insert("g.rs".into(), src.to_string());
        graph.wire_language_edges(&chunks, &sources);

        let auth = chunks.iter().find(|c| c.name == "authenticate").unwrap();
        let focus = {
            let mut f = HashMap::new();
            f.insert(graph.id_of(&auth.id()).unwrap(), 1.0);
            f
        };
        let ranked = graph.personalized_pagerank(&focus, 0.85, 40, 1e-10);
        let top_names: Vec<&str> = ranked
            .iter()
            .take(5)
            .map(|(id, _)| graph.nodes[*id].name.as_str())
            .collect();
        assert!(
            top_names.contains(&"validate_token")
                || top_names.contains(&"lookup_user")
                || top_names.contains(&"decode_jwt"),
            "expected callees near the top, got {top_names:?}"
        );
        let unrelated_rank = ranked
            .iter()
            .find(|(id, _)| graph.nodes[*id].name == "unrelated")
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        let validate_rank = ranked
            .iter()
            .find(|(id, _)| graph.nodes[*id].name == "validate_token")
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        assert!(
            validate_rank >= unrelated_rank,
            "callee should outrank an unrelated node ({validate_rank} vs {unrelated_rank})"
        );
    }

    #[test]
    fn vectorizer_ranks_paraphrase_above_noise() {
        let mut v = SemanticVectorizer::new();
        v.fit(&[
            "authenticate user token jwt",
            "render html template",
            "decode jwt token claims",
        ]);
        let q = "authenticate jwt token";
        let a = v.cosine(q, "authenticate user token jwt");
        let b = v.cosine(q, "render html template");
        assert!(a > b, "{a} vs {b}");
    }
}
