//! Overlap-reconciling, confidence-calibrated Reciprocal Rank Fusion.
//!
//! Independent ranked lists (vector search, AST symbol names, regex
//! keywords) are fused with dynamic per-ranker weights:
//!
//! ```text
//! Score = Weight * (1.0 / (K + Rank))
//! ```
//!
//! Before scoring, snippets that live in the same file and sit inside a
//! 15-line proximity window are merged into one continuous block so a
//! function and the helper immediately below it are not split across
//! context windows. An absolute high-confidence hit (`> 0.95`) bypasses
//! the positional penalty and is floated to the top of the combined list.

use std::collections::HashMap;
use std::cmp::Ordering;

/// Default RRF damping constant.
pub const DEFAULT_K: f64 = 60.0;
/// Inclusive line-distance at which two snippets in the same file merge.
pub const PROXIMITY_WINDOW: usize = 15;
/// Absolute confidence that triggers the outlier override.
pub const CONFIDENCE_OVERRIDE: f64 = 0.95;

/// One hit from a single ranker.
#[derive(Debug, Clone)]
pub struct RankedItem {
    pub id: String,
    pub file_path: String,
    pub line_start: usize,
    pub line_end: usize,
    pub content: String,
    /// 1-indexed rank inside this ranker's list.
    pub rank: usize,
    pub confidence: f64,
    pub source_ranker: String,
}

/// A complete ranked list produced by one retriever.
#[derive(Debug, Clone)]
pub struct RankedList {
    pub name: String,
    pub weight: f64,
    pub items: Vec<RankedItem>,
}

impl RankedList {
    pub fn new(name: impl Into<String>, weight: f64, mut items: Vec<RankedItem>) -> Self {
        let name = name.into();
        // Guarantee 1-indexed dense ranks if the caller left them at 0.
        for (i, item) in items.iter_mut().enumerate() {
            if item.rank == 0 {
                item.rank = i + 1;
            }
            if item.source_ranker.is_empty() {
                item.source_ranker = name.clone();
            }
        }
        Self {
            name,
            weight,
            items,
        }
    }
}

/// One fused, possibly merged, candidate.
#[derive(Debug, Clone)]
pub struct BlendedCandidate {
    pub id: String,
    pub file_path: String,
    pub line_start: usize,
    pub line_end: usize,
    pub content: String,
    pub score: f64,
    pub confidence: f64,
    pub override_floated: bool,
    pub members: Vec<String>,
    pub contributions: Vec<RankContribution>,
}

#[derive(Debug, Clone)]
pub struct RankContribution {
    pub ranker: String,
    pub weight: f64,
    pub rank: usize,
    pub confidence: f64,
    pub partial: f64,
}

/// Fusion engine.
#[derive(Debug, Clone)]
pub struct RrfBlender {
    pub k: f64,
    pub proximity_window: usize,
    pub confidence_override: f64,
}

impl Default for RrfBlender {
    fn default() -> Self {
        Self {
            k: DEFAULT_K,
            proximity_window: PROXIMITY_WINDOW,
            confidence_override: CONFIDENCE_OVERRIDE,
        }
    }
}

impl RrfBlender {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_k(mut self, k: f64) -> Self {
        self.k = k.max(0.0);
        self
    }

    /// Fuse `lists`. When `sources` contains the original file text, merged
    /// windows are reconstructed as a single continuous slice so the caller
    /// never has to stitch snippets.
    pub fn blend(
        &self,
        lists: &[RankedList],
        sources: &HashMap<String, String>,
    ) -> Vec<BlendedCandidate> {
        if lists.is_empty() {
            return Vec::new();
        }

        // Flatten into working records.
        let mut records: Vec<WorkRec> = Vec::new();
        for list in lists {
            for item in &list.items {
                records.push(WorkRec {
                    id: item.id.clone(),
                    file_path: item.file_path.clone(),
                    line_start: item.line_start.max(1),
                    line_end: item.line_end.max(item.line_start.max(1)),
                    content: item.content.clone(),
                    ranker: list.name.clone(),
                    weight: list.weight,
                    rank: item.rank.max(1),
                    confidence: item.confidence.clamp(0.0, 1.0),
                });
            }
        }
        if records.is_empty() {
            return Vec::new();
        }

        let parent = cluster_records(&records, self.proximity_window);
        let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, p) in parent.iter().enumerate() {
            let root = find_parent(&parent, *p);
            clusters.entry(root).or_default().push(i);
        }

        let mut blended = Vec::with_capacity(clusters.len());
        for members in clusters.values() {
            blended.push(self.materialize(members, &records, sources));
        }

        blended.sort_by(|a, b| compare_blended(a, b));
        blended
    }

    fn materialize(
        &self,
        members: &[usize],
        records: &[WorkRec],
        sources: &HashMap<String, String>,
    ) -> BlendedCandidate {
        let file = records[members[0]].file_path.clone();
        let mut line_start = usize::MAX;
        let mut line_end = 0usize;
        let mut max_conf: f64 = 0.0;
        let mut ids = Vec::new();
        // Best (lowest) rank per ranker, plus the weight/confidence of that hit.
        let mut best: HashMap<String, (usize, f64, f64)> = HashMap::new();

        for &i in members {
            let r = &records[i];
            line_start = line_start.min(r.line_start);
            line_end = line_end.max(r.line_end);
            max_conf = max_conf.max(r.confidence);
            if !ids.contains(&r.id) {
                ids.push(r.id.clone());
            }
            let entry = best.entry(r.ranker.clone()).or_insert((r.rank, r.weight, r.confidence));
            if r.rank < entry.0 {
                *entry = (r.rank, r.weight, r.confidence);
            }
            if r.confidence > entry.2 {
                entry.2 = r.confidence;
            }
        }

        let override_floated = max_conf > self.confidence_override;
        let mut contributions = Vec::new();
        let mut score = 0.0;
        for (ranker, (rank, weight, confidence)) in &best {
            let partial = if override_floated {
                // Bypass the positional penalty entirely.
                *weight * (1.0 + *confidence)
            } else {
                *weight * (1.0 / (self.k + *rank as f64))
            };
            score += partial;
            contributions.push(RankContribution {
                ranker: ranker.clone(),
                weight: *weight,
                rank: *rank,
                confidence: *confidence,
                partial,
            });
        }
        if override_floated {
            // Guarantee override candidates sort above ordinary RRF scores.
            score += 1_000.0 + max_conf;
        }

        let content = reconstruct(&file, line_start, line_end, members, records, sources);
        let id = if ids.len() == 1 {
            ids[0].clone()
        } else {
            format!("{}:{}-{}", file, line_start, line_end)
        };

        BlendedCandidate {
            id,
            file_path: file,
            line_start,
            line_end,
            content,
            score,
            confidence: max_conf,
            override_floated,
            members: ids,
            contributions,
        }
    }
}

#[derive(Debug, Clone)]
struct WorkRec {
    id: String,
    file_path: String,
    line_start: usize,
    line_end: usize,
    content: String,
    ranker: String,
    weight: f64,
    rank: usize,
    confidence: f64,
}

fn cluster_records(records: &[WorkRec], window: usize) -> Vec<usize> {
    let n = records.len();
    let mut parent: Vec<usize> = (0..n).collect();
    for i in 0..n {
        for j in (i + 1)..n {
            if should_merge(&records[i], &records[j], window) {
                union(&mut parent, i, j);
            }
        }
    }
    for i in 0..n {
        let _ = find_parent_mut(&mut parent, i);
    }
    parent
}

fn should_merge(a: &WorkRec, b: &WorkRec, window: usize) -> bool {
    if a.id == b.id && !a.id.is_empty() {
        return true;
    }
    if a.file_path != b.file_path {
        return false;
    }
    // Large structural scopes (a whole impl, a whole class) stay independent
    // token chunks. Only small neighbouring fragments are glued so a 15-line
    // helper sitting under its caller is not split, without letting
    // single-linkage clustering swallow an entire file.
    let span_a = a.line_end.saturating_sub(a.line_start).saturating_add(1);
    let span_b = b.line_end.saturating_sub(b.line_start).saturating_add(1);
    if span_a > window + 5 || span_b > window + 5 {
        return false;
    }
    let dist = line_distance(a.line_start, a.line_end, b.line_start, b.line_end);
    if dist > window {
        return false;
    }
    let merged_span = a.line_start.min(b.line_start).abs_diff(a.line_end.max(b.line_end)) + 1;
    merged_span <= window + span_a.max(span_b)
}

fn line_distance(a0: usize, a1: usize, b0: usize, b1: usize) -> usize {
    // Overlap or containment → distance 0.
    if a0 <= b1 && b0 <= a1 {
        return 0;
    }
    if a1 < b0 {
        b0 - a1 - 1
    } else {
        a0 - b1 - 1
    }
}

fn find_parent(parent: &[usize], mut x: usize) -> usize {
    while parent[x] != x {
        x = parent[x];
    }
    x
}

fn find_parent_mut(parent: &mut [usize], x: usize) -> usize {
    if parent[x] != x {
        parent[x] = find_parent_mut(parent, parent[x]);
    }
    parent[x]
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let ra = find_parent_mut(parent, a);
    let rb = find_parent_mut(parent, b);
    if ra != rb {
        parent[rb] = ra;
    }
}

fn reconstruct(
    file: &str,
    line_start: usize,
    line_end: usize,
    members: &[usize],
    records: &[WorkRec],
    sources: &HashMap<String, String>,
) -> String {
    if let Some(src) = sources.get(file) {
        let lines: Vec<&str> = src.lines().collect();
        let lo = line_start.saturating_sub(1).min(lines.len());
        let hi = line_end.min(lines.len());
        if lo < hi {
            return lines[lo..hi].join("\n");
        }
    }
    // Fallback: concatenate unique member contents in line order.
    let mut parts: Vec<(usize, &str)> = members
        .iter()
        .map(|&i| (records[i].line_start, records[i].content.as_str()))
        .collect();
    parts.sort_by_key(|(l, _)| *l);
    let mut seen = Vec::new();
    let mut out = String::new();
    for (_, c) in parts {
        if seen.contains(&c) {
            continue;
        }
        seen.push(c);
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(c);
    }
    out
}

fn compare_blended(a: &BlendedCandidate, b: &BlendedCandidate) -> Ordering {
    match (a.override_floated, b.override_floated) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (true, true) => b
            .confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(Ordering::Equal)
            .then_with(|| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal)),
        (false, false) => b
            .score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.line_start.cmp(&b.line_start)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: &str, file: &str, start: usize, end: usize, rank: usize, conf: f64) -> RankedItem {
        RankedItem {
            id: id.into(),
            file_path: file.into(),
            line_start: start,
            line_end: end,
            content: format!("// {id} {start}-{end}"),
            rank,
            confidence: conf,
            source_ranker: String::new(),
        }
    }

    #[test]
    fn merges_nearby_windows() {
        let vector = RankedList::new(
            "vector",
            1.0,
            vec![item("a", "f.rs", 10, 14, 1, 0.4)],
        );
        let ast = RankedList::new(
            "ast",
            1.0,
            vec![item("b", "f.rs", 16, 20, 1, 0.4)],
        );
        let src = (1..=30)
            .map(|i| format!("line{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let mut sources = HashMap::new();
        sources.insert("f.rs".into(), src);
        let out = RrfBlender::new().blend(&[vector, ast], &sources);
        assert_eq!(out.len(), 1, "nearby snippets must merge: {out:?}");
        assert_eq!(out[0].line_start, 10);
        assert_eq!(out[0].line_end, 20);
        assert!(out[0].content.contains("line10"));
        assert!(out[0].content.contains("line20"));
    }

    #[test]
    fn confidence_override_floats() {
        let vector = RankedList::new(
            "vector",
            1.0,
            vec![item("hot", "a.rs", 1, 2, 8, 0.99)],
        );
        let ast = RankedList::new(
            "ast",
            1.0,
            vec![item("cold", "b.rs", 1, 2, 1, 0.2)],
        );
        let out = RrfBlender::new().blend(&[vector, ast], &HashMap::new());
        assert_eq!(out[0].id, "hot");
        assert!(out[0].override_floated);
    }

    #[test]
    fn rrf_formula_prefers_multi_list_agreement() {
        let vector = RankedList::new(
            "vector",
            1.0,
            vec![
                item("both", "a.rs", 1, 1, 2, 0.5),
                item("only_v", "b.rs", 1, 1, 1, 0.5),
            ],
        );
        let ast = RankedList::new(
            "ast",
            1.0,
            vec![item("both", "a.rs", 1, 1, 2, 0.5)],
        );
        let out = RrfBlender::new().blend(&[vector, ast], &HashMap::new());
        assert_eq!(out[0].id, "both");
    }
}
