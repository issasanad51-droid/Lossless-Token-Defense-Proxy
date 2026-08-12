//! Custom Tree-Sitter boundary-scope mapping.
//!
//! The Cargo manifest pins the `tree-sitter` *runtime* only — language
//! grammars are not bundled. This module therefore implements the mapping
//! rules themselves: it walks source text, isolates Struct / Enum / Class /
//! Function (and Impl / Trait / Module) bodies, and emits one immutable
//! [`ScopeChunk`] per independent structural scope. Coordinates are stored
//! as `tree_sitter::Point` / `tree_sitter::Range` so the rest of the engine
//! speaks the same dialect a grammar-backed parser would.
//!
//! Chunks are **never** produced by static window slicing. A chunk exists
//! only when a language production opens and its matching closer (brace or
//! indent block) is found.

use regex::Regex;
use std::sync::OnceLock;
use tree_sitter::{Parser, Point, Range};

/// Kind of structural scope isolated by the mapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScopeKind {
    Struct,
    Enum,
    Class,
    Function,
    Method,
    Impl,
    Trait,
    Module,
}

impl ScopeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ScopeKind::Struct => "struct",
            ScopeKind::Enum => "enum",
            ScopeKind::Class => "class",
            ScopeKind::Function => "function",
            ScopeKind::Method => "method",
            ScopeKind::Impl => "impl",
            ScopeKind::Trait => "trait",
            ScopeKind::Module => "module",
        }
    }

    pub fn is_callable(self) -> bool {
        matches!(self, ScopeKind::Function | ScopeKind::Method)
    }
}

/// One immutable token-chunk candidate: a single structural scope.
#[derive(Debug, Clone)]
pub struct ScopeChunk {
    pub file_path: String,
    pub name: String,
    pub kind: ScopeKind,
    /// 1-indexed inclusive.
    pub line_start: usize,
    /// 1-indexed inclusive.
    pub line_end: usize,
    pub byte_start: usize,
    pub byte_end: usize,
    pub range: Range,
    /// Exact source slice of the scope. Owned so the chunk is immutable
    /// with respect to later filter passes on the original buffer.
    pub source: String,
}

impl ScopeChunk {
    pub fn id(&self) -> String {
        format!(
            "{}:{}:{}:{}",
            self.file_path,
            self.kind.as_str(),
            self.name,
            self.line_start
        )
    }

    pub fn start_point(&self) -> Point {
        self.range.start_point
    }

    pub fn end_point(&self) -> Point {
        self.range.end_point
    }
}

/// Detected source language for the mapping rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceLang {
    Rust,
    Python,
    JavaScript,
    Unknown,
}

impl SourceLang {
    pub fn detect(path: &str, source: &str) -> Self {
        let lower = path.to_ascii_lowercase();
        if lower.ends_with(".rs") {
            return SourceLang::Rust;
        }
        if lower.ends_with(".py") {
            return SourceLang::Python;
        }
        if lower.ends_with(".js")
            || lower.ends_with(".jsx")
            || lower.ends_with(".ts")
            || lower.ends_with(".tsx")
        {
            return SourceLang::JavaScript;
        }
        let mut rust = 0;
        let mut py = 0;
        let mut js = 0;
        for line in source.lines().take(80) {
            let t = line.trim();
            if t.starts_with("fn ")
                || t.starts_with("pub fn ")
                || t.starts_with("impl ")
                || t.starts_with("struct ")
                || t.starts_with("enum ")
            {
                rust += 1;
            }
            if t.starts_with("def ") || t.starts_with("class ") || t.starts_with("async def ") {
                py += 1;
            }
            if t.starts_with("function ") || t.starts_with("export function ") {
                js += 1;
            }
        }
        if rust >= py && rust >= js && rust > 0 {
            SourceLang::Rust
        } else if py >= js && py > 0 {
            SourceLang::Python
        } else if js > 0 {
            SourceLang::JavaScript
        } else {
            SourceLang::Unknown
        }
    }
}

/// Line-start byte index used to convert offsets into Tree-Sitter points.
#[derive(Debug, Clone)]
pub struct LineIndex {
    starts: Vec<usize>,
    len: usize,
}

impl LineIndex {
    pub fn new(src: &str) -> Self {
        let mut starts = vec![0usize];
        for (i, ch) in src.char_indices() {
            if ch == '\n' {
                starts.push(i + 1);
            }
        }
        Self {
            starts,
            len: src.len(),
        }
    }

    /// 0-indexed row/column, matching Tree-Sitter.
    pub fn point(&self, byte: usize) -> Point {
        let byte = byte.min(self.len);
        let row = match self.starts.binary_search(&byte) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        let col = byte.saturating_sub(self.starts[row]);
        Point::new(row, col)
    }

    /// 1-indexed line number that contains `byte`.
    pub fn line_of(&self, byte: usize) -> usize {
        self.point(byte).row + 1
    }

    pub fn range(&self, start: usize, end: usize) -> Range {
        Range {
            start_byte: start,
            end_byte: end,
            start_point: self.point(start),
            end_point: self.point(end),
        }
    }

    pub fn line_count(&self) -> usize {
        self.starts.len()
    }

    /// Byte length of the indexed buffer.
    pub fn source_len(&self) -> usize {
        self.len
    }
}

/// Tree-Sitter-backed boundary mapper.
pub struct AstParser {
    /// Runtime is constructed so the process owns a live Tree-Sitter parser
    /// even though grammars are supplied by the custom mapping rules below.
    runtime: Parser,
}

impl std::fmt::Debug for AstParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AstParser")
            .field("language_version", &tree_sitter::LANGUAGE_VERSION)
            .finish()
    }
}

impl AstParser {
    pub fn new() -> Self {
        Self {
            runtime: Parser::new(),
        }
    }

    /// Expose the live Tree-Sitter parser (no grammar is installed).
    pub fn runtime(&mut self) -> &mut Parser {
        &mut self.runtime
    }

    /// Isolate every independent structural scope in `source`.
    pub fn parse_file(&self, file_path: &str, source: &str) -> Vec<ScopeChunk> {
        let lang = SourceLang::detect(file_path, source);
        let index = LineIndex::new(source);
        let mut headers = match lang {
            SourceLang::Python => find_python_headers(source),
            SourceLang::JavaScript => find_js_headers(source),
            SourceLang::Rust | SourceLang::Unknown => {
                let mut h = find_rust_headers(source);
                if h.is_empty() {
                    h.extend(find_python_headers(source));
                    h.extend(find_js_headers(source));
                }
                h
            }
        };
        headers.sort_by_key(|h| h.byte_start);

        let mut chunks = Vec::with_capacity(headers.len());
        for header in headers {
            if let Some(chunk) = close_scope(file_path, source, &index, &header, lang) {
                chunks.push(chunk);
            }
        }
        chunks
    }
}

impl Default for AstParser {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
struct Header {
    kind: ScopeKind,
    name: String,
    byte_start: usize,
    /// Byte offset where the body opener (`{` or `:`) is expected to live.
    probe_from: usize,
    indent: usize,
}

fn rust_patterns() -> &'static [(ScopeKind, Regex)] {
    static CELL: OnceLock<Vec<(ScopeKind, Regex)>> = OnceLock::new();
    CELL.get_or_init(|| {
        let vis = r"(?:pub(?:\s*\([^)]*\))?\s+)?";
        vec![
            (
                ScopeKind::Function,
                Regex::new(&format!(
                    r#"(?m)^[\t ]*{vis}(?:async\s+)?(?:const\s+)?(?:unsafe\s+)?(?:extern\s+(?:"[^"]+"\s+)?)?fn\s+([A-Za-z_][A-Za-z0-9_]*)"#
                ))
                .unwrap(),
            ),
            (
                ScopeKind::Struct,
                Regex::new(&format!(
                    r"(?m)^[\t ]*{vis}struct\s+([A-Za-z_][A-Za-z0-9_]*)"
                ))
                .unwrap(),
            ),
            (
                ScopeKind::Enum,
                Regex::new(&format!(r"(?m)^[\t ]*{vis}enum\s+([A-Za-z_][A-Za-z0-9_]*)"))
                    .unwrap(),
            ),
            (
                ScopeKind::Trait,
                Regex::new(&format!(
                    r"(?m)^[\t ]*{vis}(?:unsafe\s+)?(?:auto\s+)?trait\s+([A-Za-z_][A-Za-z0-9_]*)"
                ))
                .unwrap(),
            ),
            (
                ScopeKind::Impl,
                Regex::new(&format!(
                    r"(?m)^[\t ]*{vis}(?:unsafe\s+)?impl\b(?:\s*<[^>]*>)?\s*(?:([A-Za-z_][A-Za-z0-9_:]*)\s+for\s+)?([A-Za-z_][A-Za-z0-9_:]*)"
                ))
                .unwrap(),
            ),
            (
                ScopeKind::Module,
                Regex::new(&format!(r"(?m)^[\t ]*{vis}mod\s+([A-Za-z_][A-Za-z0-9_]*)"))
                    .unwrap(),
            ),
        ]
    })
}

fn find_rust_headers(source: &str) -> Vec<Header> {
    let mut out = Vec::new();
    for (kind, re) in rust_patterns() {
        for cap in re.captures_iter(source) {
            let full = cap.get(0).unwrap();
            let name = if *kind == ScopeKind::Impl {
                cap.get(2)
                    .or_else(|| cap.get(1))
                    .map(|m| m.as_str())
                    .unwrap_or("impl")
                    .to_string()
            } else {
                cap.get(1)
                    .map(|m| m.as_str())
                    .unwrap_or("anon")
                    .to_string()
            };
            let indent = leading_indent(source, full.start());
            let kind = if *kind == ScopeKind::Function && indent > 0 {
                ScopeKind::Method
            } else {
                *kind
            };
            out.push(Header {
                kind,
                name,
                byte_start: full.start(),
                probe_from: full.end(),
                indent,
            });
        }
    }
    out
}

fn python_patterns() -> &'static [(ScopeKind, Regex)] {
    static CELL: OnceLock<Vec<(ScopeKind, Regex)>> = OnceLock::new();
    CELL.get_or_init(|| {
        vec![
            (
                ScopeKind::Function,
                Regex::new(r"(?m)^([ \t]*)(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)").unwrap(),
            ),
            (
                ScopeKind::Class,
                Regex::new(r"(?m)^([ \t]*)class\s+([A-Za-z_][A-Za-z0-9_]*)").unwrap(),
            ),
        ]
    })
}

fn find_python_headers(source: &str) -> Vec<Header> {
    let mut out = Vec::new();
    for (kind, re) in python_patterns() {
        for cap in re.captures_iter(source) {
            let full = cap.get(0).unwrap();
            let indent = cap
                .get(1)
                .map(|m| m.as_str().chars().count())
                .unwrap_or(0);
            let name = cap
                .get(2)
                .map(|m| m.as_str())
                .unwrap_or("anon")
                .to_string();
            let kind = if *kind == ScopeKind::Function && indent > 0 {
                ScopeKind::Method
            } else {
                *kind
            };
            // Decorators immediately above become part of the scope.
            let byte_start = extend_python_decorators(source, full.start(), indent);
            out.push(Header {
                kind,
                name,
                byte_start,
                probe_from: full.end(),
                indent,
            });
        }
    }
    out
}

fn extend_python_decorators(source: &str, def_start: usize, indent: usize) -> usize {
    let bytes = source.as_bytes();
    let mut cursor = def_start;
    loop {
        if cursor == 0 {
            break;
        }
        // Walk back one line.
        let mut line_end = cursor;
        if line_end > 0 && bytes[line_end - 1] == b'\n' {
            line_end -= 1;
        }
        let mut line_start = line_end;
        while line_start > 0 && bytes[line_start - 1] != b'\n' {
            line_start -= 1;
        }
        let line = &source[line_start..line_end];
        let trimmed = line.trim();
        if trimmed.is_empty() {
            cursor = line_start;
            continue;
        }
        let this_indent = line.chars().take_while(|c| *c == ' ' || *c == '\t').count();
        if this_indent == indent && trimmed.starts_with('@') {
            cursor = line_start;
            continue;
        }
        break;
    }
    cursor
}

fn js_patterns() -> &'static [(ScopeKind, Regex)] {
    static CELL: OnceLock<Vec<(ScopeKind, Regex)>> = OnceLock::new();
    CELL.get_or_init(|| {
        vec![
            (
                ScopeKind::Function,
                Regex::new(
                    r"(?m)^[\t ]*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)",
                )
                .unwrap(),
            ),
            (
                ScopeKind::Class,
                Regex::new(r"(?m)^[\t ]*(?:export\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)").unwrap(),
            ),
            (
                ScopeKind::Function,
                Regex::new(
                    r"(?m)^[\t ]*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_][A-Za-z0-9_]*)\s*=>",
                )
                .unwrap(),
            ),
        ]
    })
}

fn find_js_headers(source: &str) -> Vec<Header> {
    let mut out = Vec::new();
    for (kind, re) in js_patterns() {
        for cap in re.captures_iter(source) {
            let full = cap.get(0).unwrap();
            let name = cap
                .get(1)
                .map(|m| m.as_str())
                .unwrap_or("anon")
                .to_string();
            out.push(Header {
                kind: *kind,
                name,
                byte_start: full.start(),
                probe_from: full.end(),
                indent: leading_indent(source, full.start()),
            });
        }
    }
    out
}

fn leading_indent(source: &str, at: usize) -> usize {
    let line_start = source[..at].rfind('\n').map(|i| i + 1).unwrap_or(0);
    source[line_start..at]
        .chars()
        .take_while(|c| *c == ' ' || *c == '\t')
        .count()
}

fn close_scope(
    file_path: &str,
    source: &str,
    index: &LineIndex,
    header: &Header,
    lang: SourceLang,
) -> Option<ScopeChunk> {
    let (end, _opener) = match lang {
        SourceLang::Python => close_python_block(source, header)?,
        _ => close_brace_block(source, header.probe_from)
            .or_else(|| close_python_block(source, header))?,
    };
    if end <= header.byte_start {
        return None;
    }
    let source_slice = source[header.byte_start..end].to_string();
    Some(ScopeChunk {
        file_path: file_path.to_string(),
        name: header.name.clone(),
        kind: header.kind,
        line_start: index.line_of(header.byte_start),
        line_end: index.line_of(end.saturating_sub(1)),
        byte_start: header.byte_start,
        byte_end: end,
        range: index.range(header.byte_start, end),
        source: source_slice,
    })
}

/// Scan from `probe` for `{` (or `;` for unit structs) then match braces,
/// ignoring braces that live inside strings or comments.
fn close_brace_block(source: &str, probe: usize) -> Option<(usize, usize)> {
    let b = source.as_bytes();
    let mut i = probe;
    let mut in_str: Option<u8> = None;
    let mut escaped = false;
    let mut line_comment = false;
    let mut block_comment = false;

    // Find opener or semicolon.
    let mut opener = None;
    while i < b.len() {
        let c = b[i];
        if line_comment {
            if c == b'\n' {
                line_comment = false;
            }
            i += 1;
            continue;
        }
        if block_comment {
            if c == b'*' && i + 1 < b.len() && b[i + 1] == b'/' {
                block_comment = false;
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }
        if let Some(d) = in_str {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == d {
                in_str = None;
            }
            i += 1;
            continue;
        }
        if c == b'"' || c == b'\'' {
            in_str = Some(c);
            i += 1;
            continue;
        }
        if c == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            line_comment = true;
            i += 2;
            continue;
        }
        if c == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            block_comment = true;
            i += 2;
            continue;
        }
        if c == b'{' {
            opener = Some(i);
            break;
        }
        if c == b';' {
            return Some((i + 1, i));
        }
        i += 1;
    }
    let opener = opener?;
    i = opener + 1;
    let mut depth = 1i32;
    in_str = None;
    escaped = false;
    line_comment = false;
    block_comment = false;
    while i < b.len() {
        let c = b[i];
        if line_comment {
            if c == b'\n' {
                line_comment = false;
            }
            i += 1;
            continue;
        }
        if block_comment {
            if c == b'*' && i + 1 < b.len() && b[i + 1] == b'/' {
                block_comment = false;
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }
        if let Some(d) = in_str {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == d {
                in_str = None;
            }
            i += 1;
            continue;
        }
        if c == b'"' || c == b'\'' {
            in_str = Some(c);
            i += 1;
            continue;
        }
        if c == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            line_comment = true;
            i += 2;
            continue;
        }
        if c == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            block_comment = true;
            i += 2;
            continue;
        }
        if c == b'{' {
            depth += 1;
        } else if c == b'}' {
            depth -= 1;
            if depth == 0 {
                return Some((i + 1, opener));
            }
        }
        i += 1;
    }
    None
}

fn close_python_block(source: &str, header: &Header) -> Option<(usize, usize)> {
    let b = source.as_bytes();
    // Find the `:` that terminates the header (paren-aware, multi-line sigs).
    let mut i = header.probe_from;
    let mut paren = 0i32;
    let mut in_str: Option<u8> = None;
    let mut escaped = false;
    let mut colon = None;
    while i < b.len() {
        let c = b[i];
        if let Some(d) = in_str {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == d {
                in_str = None;
            }
            i += 1;
            continue;
        }
        if c == b'"' || c == b'\'' {
            in_str = Some(c);
            i += 1;
            continue;
        }
        if c == b'(' || c == b'[' || c == b'{' {
            paren += 1;
        } else if c == b')' || c == b']' || c == b'}' {
            paren -= 1;
        } else if c == b':' && paren == 0 {
            colon = Some(i);
            break;
        }
        i += 1;
    }
    let colon = colon?;
    // Consume the rest of the header line.
    let mut line_end = colon;
    while line_end < b.len() && b[line_end] != b'\n' {
        line_end += 1;
    }
    if line_end < b.len() {
        line_end += 1; // include newline
    }
    // Body: subsequent lines whose indent is strictly greater, plus blank
    // lines that sit inside the block.
    let mut end = line_end;
    let mut scan = line_end;
    while scan < b.len() {
        let row_start = scan;
        while scan < b.len() && b[scan] != b'\n' {
            scan += 1;
        }
        let row_end = scan;
        if scan < b.len() {
            scan += 1;
        }
        let row = &source[row_start..row_end];
        if row.trim().is_empty() {
            end = scan;
            continue;
        }
        let indent = row.chars().take_while(|c| *c == ' ' || *c == '\t').count();
        if indent > header.indent {
            end = scan;
        } else {
            break;
        }
    }
    Some((end.max(line_end), colon))
}

/// Extract `use` / `import` identifiers from a file (used by the PPR graph).
pub fn extract_imports(source: &str) -> Vec<String> {
    static USE_RE: OnceLock<Regex> = OnceLock::new();
    static IMPORT_RE: OnceLock<Regex> = OnceLock::new();
    let use_re = USE_RE.get_or_init(|| {
        Regex::new(r"(?m)^\s*(?:pub\s+)?use\s+([A-Za-z0-9_:,\{\}\s]+);").unwrap()
    });
    let import_re = IMPORT_RE.get_or_init(|| {
        Regex::new(r"(?m)^\s*(?:from\s+([A-Za-z0-9_\.]+)\s+import\s+([A-Za-z0-9_\*,\s]+)|import\s+([A-Za-z0-9_\.]+))")
            .unwrap()
    });
    let mut names = Vec::new();
    for cap in use_re.captures_iter(source) {
        if let Some(m) = cap.get(1) {
            for tok in m.as_str().split(|c: char| {
                c == ':' || c == '{' || c == '}' || c == ',' || c == ' ' || c == '\n'
            }) {
                let t = tok.trim();
                if !t.is_empty() && t != "self" && t != "super" && t != "crate" {
                    names.push(t.to_string());
                }
            }
        }
    }
    for cap in import_re.captures_iter(source) {
        if let Some(m) = cap.get(2) {
            for tok in m.as_str().split(',') {
                let t = tok.trim();
                if !t.is_empty() && t != "*" {
                    names.push(t.split_whitespace().last().unwrap_or(t).to_string());
                }
            }
        }
        if let Some(m) = cap.get(1) {
            if let Some(last) = m.as_str().split('.').last() {
                names.push(last.to_string());
            }
        }
        if let Some(m) = cap.get(3) {
            if let Some(last) = m.as_str().split('.').last() {
                names.push(last.to_string());
            }
        }
    }
    names.sort();
    names.dedup();
    names
}

/// True when `haystack` contains a call-site of `name`.
///
/// Definition sites (`fn name(`, `class name(`, `struct name`) are ignored
/// so a type does not appear to call itself.
pub fn contains_call(haystack: &str, name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    static CACHE: OnceLock<std::sync::Mutex<std::collections::HashMap<String, Regex>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
    let re = guard.entry(name.to_string()).or_insert_with(|| {
        let pat = format!(r"(?m)(?:^|[^\w]){}[\t ]*(?:\(|::)", regex::escape(name));
        Regex::new(&pat).unwrap_or_else(|_| Regex::new(r"$^").unwrap())
    });
    for m in re.find_iter(haystack) {
        if !is_definition_prefix(haystack, m.start()) {
            return true;
        }
    }
    false
}

fn is_definition_prefix(src: &str, at: usize) -> bool {
    let line_start = src[..at].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let prefix = src[line_start..at].trim_end();
    let last = prefix
        .split(|c: char| c.is_whitespace() || c == ':' || c == '>')
        .last()
        .unwrap_or("");
    matches!(
        last,
        "def" | "class" | "fn" | "struct" | "enum" | "trait" | "mod" | "impl"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_rust_struct_enum_fn() {
        let src = r#"
pub struct User { pub id: u64 }
pub enum AuthError { Denied, Expired }
pub fn authenticate(tok: &str) -> bool {
    validate_token(tok)
}
fn validate_token(tok: &str) -> bool { !tok.is_empty() }
"#;
        let chunks = AstParser::new().parse_file("auth.rs", src);
        let names: Vec<_> = chunks.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"User"));
        assert!(names.contains(&"AuthError"));
        assert!(names.contains(&"authenticate"));
        assert!(names.contains(&"validate_token"));
        let auth = chunks.iter().find(|c| c.name == "authenticate").unwrap();
        assert!(auth.source.contains("validate_token"));
        assert!(auth.line_start >= 1);
        assert!(auth.line_end >= auth.line_start);
    }

    #[test]
    fn extracts_python_class_and_methods() {
        let src = "class Auth:\n    def login(self):\n        return self.ok()\n    def ok(self):\n        return True\n";
        let chunks = AstParser::new().parse_file("a.py", src);
        let names: Vec<_> = chunks.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"Auth"));
        assert!(names.contains(&"login"));
        assert!(names.contains(&"ok"));
    }

    #[test]
    fn never_window_slices() {
        let src = "fn a() {\n    let x = 1;\n}\nfn b() {\n    let y = 2;\n}\n";
        let chunks = AstParser::new().parse_file("t.rs", src);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].source.contains("fn a"));
        assert!(!chunks[0].source.contains("fn b"));
    }
}
