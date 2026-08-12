//! # Subsystem A - AST-bounded vector chunking
//!
//! Sliding character windows are the default in most retrieval stacks and they
//! are wrong for code: a 512-token window cuts a function in half, so the model
//! receives a signature with no body, or a body with no signature, and both
//! halves get embedded as if they were whole thoughts. Worse, overlapping
//! windows re-send the same lines several times - pure token waste.
//!
//! This module instead extracts **explicit logical scopes** - structs, enums,
//! traits, classes, functions, methods, impl blocks - and treats each one as
//! exactly one immutable chunk candidate with hard physical coordinates
//! (`file_path`, `line_start`, `line_end`). A chunk is never a fragment.
//!
//! ## Why hand-written scanners instead of tree-sitter
//!
//! Full grammars are overkill for "where does this scope start and end", and
//! they drag in native build steps. The approach here is a two-phase scanner:
//!
//! 1. **Mask pass** ([`mask_source`]) - blanks out every string literal and
//!    comment, preserving byte offsets and newlines. This runs *first*, so a
//!    brace inside `"}"` or `// }` can never corrupt a scope boundary. That one
//!    bug is what makes naive brace-counting indexers silently useless.
//! 2. **Scope pass** - walks the masked text, matches per-language declaration
//!    patterns, then finds the extent by brace balance (Rust/JS/TS/Go) or by
//!    indentation (Python).
//!
//! The result is deterministic, dependency-free, and fast enough to re-index on
//! every request.

use std::collections::{HashMap, HashSet};

use super::{Filter, PipelineContext, PipelineError};

/// Dimensionality of the mock embedding space.
pub const EMBEDDING_DIM: usize = 256;

/// Languages the scanner understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
}

impl Language {
    /// Infer language from a file extension. Unknown extensions are skipped
    /// rather than guessed - a wrong guess produces garbage scopes.
    pub fn from_path(path: &str) -> Option<Language> {
        let ext = path.rsplit('.').next()?;
        match ext {
            "rs" => Some(Language::Rust),
            "py" | "pyi" => Some(Language::Python),
            "js" | "jsx" | "mjs" | "cjs" => Some(Language::JavaScript),
            "ts" | "tsx" | "mts" | "cts" => Some(Language::TypeScript),
            "go" => Some(Language::Go),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Language::Rust => "rust",
            Language::Python => "python",
            Language::JavaScript => "javascript",
            Language::TypeScript => "typescript",
            Language::Go => "go",
        }
    }

    /// Python delimits scopes by indentation; the rest use braces.
    fn is_brace_delimited(&self) -> bool {
        !matches!(self, Language::Python)
    }

    /// Line-comment markers.
    fn line_comment(&self) -> &'static str {
        match self {
            Language::Python => "#",
            _ => "//",
        }
    }
}

/// The kind of structural scope a chunk represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScopeKind {
    Function,
    Method,
    Struct,
    Enum,
    Class,
    Trait,
    Impl,
    Interface,
    TypeAlias,
    Module,
}

impl ScopeKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            ScopeKind::Function => "function",
            ScopeKind::Method => "method",
            ScopeKind::Struct => "struct",
            ScopeKind::Enum => "enum",
            ScopeKind::Class => "class",
            ScopeKind::Trait => "trait",
            ScopeKind::Impl => "impl",
            ScopeKind::Interface => "interface",
            ScopeKind::TypeAlias => "type_alias",
            ScopeKind::Module => "module",
        }
    }

    /// Callable scopes are the useful retrieval targets; containers mostly
    /// exist to give callables a parent.
    pub fn is_callable(&self) -> bool {
        matches!(self, ScopeKind::Function | ScopeKind::Method)
    }
}

/// One immutable chunk candidate: a complete logical scope plus its exact
/// physical coordinates and its embedding.
#[derive(Debug, Clone)]
pub struct CodeChunk {
    /// Stable identity, `file_path::qualified_name`.
    pub id: String,
    pub file_path: String,
    pub language: Language,
    pub kind: ScopeKind,
    /// Bare identifier, e.g. `charge`.
    pub name: String,
    /// Parent-qualified identifier, e.g. `PaymentProcessor.charge`.
    pub qualified_name: String,
    /// 1-based, inclusive.
    pub line_start: usize,
    /// 1-based, inclusive.
    pub line_end: usize,
    /// Verbatim source of the scope - never a fragment.
    pub source: String,
    /// Declaration line, used for cheap symbol matching.
    pub signature: String,
    /// Chunk id of the enclosing scope, if any.
    pub parent: Option<String>,
    /// Identifiers this scope references, for call-graph construction.
    pub referenced_symbols: Vec<String>,
    /// L2-normalized embedding.
    pub embedding: Vec<f32>,
}

impl CodeChunk {
    pub fn line_count(&self) -> usize {
        self.line_end.saturating_sub(self.line_start) + 1
    }

    /// True when two chunks cover any of the same lines in the same file.
    pub fn overlaps(&self, other: &CodeChunk) -> bool {
        self.file_path == other.file_path
            && self.line_start <= other.line_end
            && other.line_start <= self.line_end
    }

    /// Text an embedder should see: coordinates, signature, then body.
    pub fn embedding_text(&self) -> String {
        format!(
            "{} {} {} in {}\n{}",
            self.kind.as_str(),
            self.qualified_name,
            self.signature,
            self.file_path,
            self.source
        )
    }
}

/// A file handed to the chunker.
#[derive(Debug, Clone)]
pub struct SourceFile {
    pub path: String,
    pub content: String,
}

impl SourceFile {
    pub fn new<P: Into<String>, C: Into<String>>(path: P, content: C) -> Self {
        SourceFile {
            path: path.into(),
            content: content.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 1: masking
// ---------------------------------------------------------------------------

/// Replace every string literal and comment with spaces, preserving byte
/// offsets and newline positions.
///
/// Handles the literal forms that actually break naive scanners:
/// Rust raw strings (`r#"..."#`), lifetimes vs char literals (`'a` vs `'}'`),
/// Python triple quotes, JS template literals, Go raw backtick strings, and
/// nested block comments (legal in Rust).
///
/// Returned string has identical length and identical newline placement, so any
/// index into it maps 1:1 onto the original.
pub fn mask_source(src: &str, lang: Language) -> String {
    let bytes = src.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0usize;

    // Copy a byte, or blank it while keeping newlines intact.
    macro_rules! blank {
        ($n:expr) => {{
            for k in 0..$n {
                let b = bytes[i + k];
                out.push(if b == b'\n' { b'\n' } else { b' ' });
            }
            i += $n;
        }};
    }

    while i < bytes.len() {
        let rest = &src[i..];

        // ---- comments -----------------------------------------------------
        if lang == Language::Python {
            if bytes[i] == b'#' {
                let len = rest.find('\n').unwrap_or(rest.len());
                blank!(len);
                continue;
            }
        } else {
            if rest.starts_with("//") {
                let len = rest.find('\n').unwrap_or(rest.len());
                blank!(len);
                continue;
            }
            if rest.starts_with("/*") {
                // Rust allows nesting; C-family does not. Counting depth is
                // correct for Rust and harmless elsewhere in practice.
                let mut depth = 0usize;
                let mut j = i;
                while j < bytes.len() {
                    if src[j..].starts_with("/*") {
                        depth += 1;
                        j += 2;
                    } else if src[j..].starts_with("*/") {
                        depth -= 1;
                        j += 2;
                        if depth == 0 {
                            break;
                        }
                    } else {
                        j += 1;
                    }
                }
                let len = j - i;
                blank!(len);
                continue;
            }
        }

        // ---- Rust raw strings: r"...", r#"..."#, br#"..."# -----------------
        if lang == Language::Rust {
            let mut probe = i;
            if bytes[probe] == b'b' {
                probe += 1;
            }
            if probe < bytes.len() && bytes[probe] == b'r' {
                let mut hashes = 0usize;
                let mut k = probe + 1;
                while k < bytes.len() && bytes[k] == b'#' {
                    hashes += 1;
                    k += 1;
                }
                if k < bytes.len() && bytes[k] == b'"' {
                    // Only a raw string if `r` is not part of an identifier.
                    let prev_ok = i == 0
                        || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
                    if prev_ok {
                        let terminator = format!("\"{}", "#".repeat(hashes));
                        let body_start = k + 1;
                        let end = src[body_start..]
                            .find(&terminator)
                            .map(|p| body_start + p + terminator.len())
                            .unwrap_or(bytes.len());
                        let len = end - i;
                        blank!(len);
                        continue;
                    }
                }
            }
        }

        // ---- Go raw strings ------------------------------------------------
        if lang == Language::Go && bytes[i] == b'`' {
            let end = src[i + 1..]
                .find('`')
                .map(|p| i + 1 + p + 1)
                .unwrap_or(bytes.len());
            let len = end - i;
            blank!(len);
            continue;
        }

        // ---- JS/TS template literals ---------------------------------------
        if matches!(lang, Language::JavaScript | Language::TypeScript) && bytes[i] == b'`' {
            let mut j = i + 1;
            while j < bytes.len() {
                match bytes[j] {
                    b'\\' => j += 2,
                    b'`' => {
                        j += 1;
                        break;
                    }
                    _ => j += 1,
                }
            }
            let len = j.min(bytes.len()) - i;
            blank!(len);
            continue;
        }

        // ---- Python triple-quoted strings -----------------------------------
        if lang == Language::Python && (rest.starts_with("\"\"\"") || rest.starts_with("'''")) {
            let quote = &rest[..3];
            let end = rest[3..]
                .find(quote)
                .map(|p| i + 3 + p + 3)
                .unwrap_or(bytes.len());
            let len = end - i;
            blank!(len);
            continue;
        }

        // ---- Rust char literals and lifetimes --------------------------------
        if lang == Language::Rust && bytes[i] == b'\'' {
            // `'a` (lifetime) has no closing quote; `'}'` does. Distinguish by
            // looking ahead, otherwise a lifetime swallows the rest of the file.
            let is_char_literal = if i + 2 < bytes.len() && bytes[i + 1] == b'\\' {
                true
            } else {
                i + 2 < bytes.len() && bytes[i + 2] == b'\''
            };
            if is_char_literal {
                let mut j = i + 1;
                while j < bytes.len() {
                    match bytes[j] {
                        b'\\' => j += 2,
                        b'\'' => {
                            j += 1;
                            break;
                        }
                        _ => j += 1,
                    }
                }
                let len = j.min(bytes.len()) - i;
                blank!(len);
                continue;
            }
            out.push(bytes[i]);
            i += 1;
            continue;
        }

        // ---- ordinary quoted strings ------------------------------------------
        if bytes[i] == b'"' || bytes[i] == b'\'' {
            let quote = bytes[i];
            let mut j = i + 1;
            while j < bytes.len() {
                match bytes[j] {
                    b'\\' => j += 2,
                    b'\n' => break, // unterminated; do not run past the line
                    b if b == quote => {
                        j += 1;
                        break;
                    }
                    _ => j += 1,
                }
            }
            let len = j.min(bytes.len()) - i;
            blank!(len);
            continue;
        }

        out.push(bytes[i]);
        i += 1;
    }

    // Every byte was either copied or replaced by a same-width ASCII byte, and
    // multi-byte UTF-8 sequences are only ever copied wholesale.
    String::from_utf8(out).unwrap_or_else(|_| src.to_string())
}

// ---------------------------------------------------------------------------
// Phase 2: scope extraction
// ---------------------------------------------------------------------------

/// A declaration spotted on a line, before its extent is known.
#[derive(Debug, Clone)]
struct Declaration {
    kind: ScopeKind,
    name: String,
    line_index: usize,
    indent: usize,
}

/// Strip a leading `pub`, `pub(crate)`, `export`, `default`, `async`, etc.
fn strip_modifiers<'a>(mut line: &'a str, lang: Language) -> &'a str {
    let modifiers: &[&str] = match lang {
        Language::Rust => &["pub(crate)", "pub(super)", "pub(self)", "pub", "async", "unsafe", "const", "extern", "default"],
        Language::JavaScript | Language::TypeScript => &["export", "default", "async", "declare", "abstract", "public", "private", "protected", "static", "readonly"],
        Language::Python => &["async"],
        Language::Go => &[],
    };
    loop {
        let trimmed = line.trim_start();
        let mut advanced = false;
        for m in modifiers {
            if let Some(rest) = trimmed.strip_prefix(m) {
                let boundary = rest
                    .chars()
                    .next()
                    .map(|c| !c.is_alphanumeric() && c != '_')
                    .unwrap_or(true);
                if boundary {
                    line = rest;
                    advanced = true;
                    break;
                }
            }
        }
        if !advanced {
            return line.trim_start();
        }
    }
}

/// Read an identifier starting at the beginning of `s`.
fn take_identifier(s: &str) -> Option<String> {
    let s = s.trim_start();
    let mut end = 0usize;
    for (idx, ch) in s.char_indices() {
        if ch.is_alphanumeric() || ch == '_' || ch == '$' {
            end = idx + ch.len_utf8();
        } else {
            break;
        }
    }
    if end == 0 {
        return None;
    }
    let ident = &s[..end];
    if ident.chars().next().map(|c| c.is_numeric()).unwrap_or(true) {
        return None;
    }
    Some(ident.to_string())
}

/// Match `keyword <identifier>` at the start of a (modifier-stripped) line.
fn match_keyword_decl(line: &str, keyword: &str) -> Option<String> {
    let rest = line.strip_prefix(keyword)?;
    let next = rest.chars().next()?;
    if next.is_alphanumeric() || next == '_' {
        return None; // `structure` is not `struct`
    }
    // Skip Rust generics on the keyword itself (`impl<T> Foo`).
    take_identifier(rest)
}

fn indent_width(line: &str) -> usize {
    let mut width = 0usize;
    for ch in line.chars() {
        match ch {
            ' ' => width += 1,
            '\t' => width += 4,
            _ => break,
        }
    }
    width
}

/// Spot a declaration on one masked line.
fn detect_declaration(line: &str, lang: Language, line_index: usize) -> Option<Declaration> {
    let indent = indent_width(line);
    let stripped = strip_modifiers(line, lang);
    if stripped.is_empty() {
        return None;
    }

    let mk = |kind: ScopeKind, name: String| {
        Some(Declaration {
            kind,
            name,
            line_index,
            indent,
        })
    };

    match lang {
        Language::Rust => {
            if let Some(n) = match_keyword_decl(stripped, "fn") {
                return mk(ScopeKind::Function, n);
            }
            if let Some(n) = match_keyword_decl(stripped, "struct") {
                return mk(ScopeKind::Struct, n);
            }
            if let Some(n) = match_keyword_decl(stripped, "enum") {
                return mk(ScopeKind::Enum, n);
            }
            if let Some(n) = match_keyword_decl(stripped, "trait") {
                return mk(ScopeKind::Trait, n);
            }
            if let Some(n) = match_keyword_decl(stripped, "mod") {
                // Only inline modules open a scope; `mod foo;` does not.
                if stripped.contains('{') {
                    return mk(ScopeKind::Module, n);
                }
                return None;
            }
            if stripped.starts_with("impl") {
                // `impl Trait for Type` -> index under Type, which is what a
                // reader searches for.
                let rest = stripped.strip_prefix("impl").unwrap_or("");
                let boundary = rest
                    .chars()
                    .next()
                    .map(|c| !c.is_alphanumeric() && c != '_')
                    .unwrap_or(false);
                if boundary {
                    let head = rest.split('{').next().unwrap_or(rest);
                    let target = head.split(" for ").last().unwrap_or(head);
                    let cleaned = target
                        .trim()
                        .trim_start_matches('<')
                        .split(|c: char| c == '<' || c == '(' || c.is_whitespace())
                        .find(|s| !s.is_empty())
                        .unwrap_or("impl");
                    return mk(ScopeKind::Impl, cleaned.to_string());
                }
            }
            None
        }
        Language::Python => {
            if let Some(n) = match_keyword_decl(stripped, "def") {
                return mk(ScopeKind::Function, n);
            }
            if let Some(n) = match_keyword_decl(stripped, "class") {
                return mk(ScopeKind::Class, n);
            }
            None
        }
        Language::Go => {
            // Methods carry a receiver: `func (p *Pool) Get() int {`.
            // This must be checked *before* `match_keyword_decl`, which looks
            // for an identifier right after the keyword and finds `(` instead,
            // so every method would otherwise be silently skipped.
            if stripped.starts_with("func (") {
                let after = stripped.splitn(2, ')').nth(1).unwrap_or("");
                if let Some(name) = take_identifier(after) {
                    return mk(ScopeKind::Method, name);
                }
            }
            if let Some(n) = match_keyword_decl(stripped, "func") {
                return mk(ScopeKind::Function, n);
            }
            if let Some(n) = match_keyword_decl(stripped, "type") {
                let tail = stripped.splitn(2, n.as_str()).nth(1).unwrap_or("");
                if tail.contains("struct") {
                    return mk(ScopeKind::Struct, n);
                }
                if tail.contains("interface") {
                    return mk(ScopeKind::Interface, n);
                }
                return mk(ScopeKind::TypeAlias, n);
            }
            None
        }
        Language::JavaScript | Language::TypeScript => {
            if let Some(n) = match_keyword_decl(stripped, "function") {
                return mk(ScopeKind::Function, n);
            }
            if let Some(n) = match_keyword_decl(stripped, "class") {
                return mk(ScopeKind::Class, n);
            }
            if lang == Language::TypeScript {
                if let Some(n) = match_keyword_decl(stripped, "interface") {
                    return mk(ScopeKind::Interface, n);
                }
                if let Some(n) = match_keyword_decl(stripped, "enum") {
                    return mk(ScopeKind::Enum, n);
                }
            }
            // `const handler = (req, res) => {` / `const f = function () {`
            for kw in ["const", "let", "var"] {
                if let Some(name) = match_keyword_decl(stripped, kw) {
                    let tail = stripped.splitn(2, '=').nth(1).unwrap_or("");
                    if tail.contains("=>") || tail.trim_start().starts_with("function") {
                        return mk(ScopeKind::Function, name);
                    }
                }
            }
            // Class methods: `foo(a, b) {` with no leading keyword.
            if stripped.contains('(') && stripped.trim_end().ends_with('{') {
                let head = stripped.split('(').next().unwrap_or("");
                let head_trim = head.trim();
                let is_control = matches!(
                    head_trim,
                    "if" | "for" | "while" | "switch" | "catch" | "do" | "else" | "try" | "with"
                );
                if !is_control && !head_trim.is_empty() {
                    if let Some(name) = take_identifier(head_trim) {
                        if name.len() == head_trim.len() {
                            return mk(ScopeKind::Method, name);
                        }
                    }
                }
            }
            None
        }
    }
}

/// Find the line index where a brace-delimited scope ends.
///
/// Starts counting from the declaration line, so a body opening on a later line
/// (`fn f(\n  a: u32,\n) {`) is still handled. Operates on masked text, so
/// braces in strings and comments cannot skew the balance.
fn find_brace_scope_end(masked_lines: &[&str], start: usize) -> Option<usize> {
    let mut depth: i64 = 0;
    let mut seen_open = false;

    for (offset, line) in masked_lines.iter().enumerate().skip(start) {
        for ch in line.chars() {
            match ch {
                '{' => {
                    depth += 1;
                    seen_open = true;
                }
                '}' => {
                    depth -= 1;
                    if seen_open && depth <= 0 {
                        return Some(offset);
                    }
                }
                _ => {}
            }
        }
        // A declaration terminated by `;` before any `{` has no body
        // (`fn f();` in a trait, `type X = Y;`).
        if !seen_open && line.trim_end().ends_with(';') {
            return Some(offset);
        }
        // Guard against a stray unbalanced brace eating the whole file.
        if !seen_open && offset > start + 40 {
            return Some(offset);
        }
    }
    if seen_open {
        Some(masked_lines.len().saturating_sub(1))
    } else {
        None
    }
}

/// Find the line index where an indentation-delimited (Python) scope ends.
fn find_indent_scope_end(masked_lines: &[&str], start: usize, decl_indent: usize) -> usize {
    let mut end = start;
    for (offset, line) in masked_lines.iter().enumerate().skip(start + 1) {
        if line.trim().is_empty() {
            continue; // blank lines belong to whatever follows
        }
        if indent_width(line) <= decl_indent {
            break;
        }
        end = offset;
    }
    end
}

/// Extract identifiers a scope references, for call-graph edges.
fn extract_referenced_symbols(masked_body: &str, own_name: &str) -> Vec<String> {
    let mut seen: Vec<String> = Vec::new();
    let bytes = masked_body.as_bytes();
    let mut i = 0usize;

    while i < bytes.len() {
        let ch = bytes[i] as char;
        if ch.is_alphabetic() || ch == '_' {
            let start = i;
            while i < bytes.len() {
                let c = bytes[i] as char;
                if c.is_alphanumeric() || c == '_' {
                    i += 1;
                } else {
                    break;
                }
            }
            let ident = &masked_body[start..i];

            // Only count it as a reference if a call parenthesis follows.
            let mut j = i;
            while j < bytes.len() && (bytes[j] as char).is_whitespace() {
                j += 1;
            }
            let is_call = j < bytes.len() && bytes[j] == b'(';

            if is_call && ident != own_name && !is_language_keyword(ident) && ident.len() > 1 {
                if !seen.iter().any(|s| s == ident) {
                    seen.push(ident.to_string());
                }
            }
            continue;
        }
        i += 1;
    }
    seen
}

fn is_language_keyword(word: &str) -> bool {
    matches!(
        word,
        "if" | "else" | "for" | "while" | "return" | "match" | "let" | "const" | "var"
            | "fn" | "def" | "class" | "struct" | "enum" | "impl" | "trait" | "pub" | "use"
            | "mod" | "func" | "type" | "interface" | "package" | "import" | "from" | "as"
            | "in" | "is" | "not" | "and" | "or" | "try" | "catch" | "except" | "finally"
            | "with" | "switch" | "case" | "default" | "break" | "continue" | "new" | "self"
            | "this" | "super" | "async" | "await" | "yield" | "throw" | "raise" | "assert"
            | "print" | "range" | "len" | "unwrap" | "expect" | "clone" | "to_string"
            | "push" | "iter" | "collect" | "map" | "filter" | "unwrap_or" | "some" | "none"
            | "ok" | "err" | "string" | "vec" | "box" | "option" | "result" | "go" | "defer"
            | "elif" | "lambda" | "pass" | "del" | "global" | "nonlocal" | "do" | "static"
    )
}

// ---------------------------------------------------------------------------
// Mock embedding
// ---------------------------------------------------------------------------

/// Deterministic local embedding.
///
/// **This is a hashed bag-of-features vector, not a learned semantic model.**
/// It behaves correctly under cosine similarity - identical text scores 1.0,
/// shared identifiers pull vectors together, unrelated code scores near 0 - so
/// the surrounding pipeline is exercised honestly. It does not capture meaning:
/// `fetch_user` and `retrieve_account` are unrelated to it. Swap this function
/// for a real embedding service and nothing downstream changes.
///
/// Identifiers are split on `snake_case` / `camelCase` and both the compound
/// and its parts are hashed, so `parse_config` partially matches `config`.
pub fn compute_local_embedding(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0f32; EMBEDDING_DIM];

    for token in tokenize_for_embedding(text) {
        // Two independent hashes per token: reduces collision artefacts in a
        // fixed-width space without needing a real vocabulary.
        let h1 = fnv1a(token.as_bytes());
        let h2 = fnv1a_seeded(token.as_bytes(), 0x9e37_79b9_7f4a_7c15);

        let i1 = (h1 % EMBEDDING_DIM as u64) as usize;
        let i2 = (h2 % EMBEDDING_DIM as u64) as usize;

        // Signed contributions keep the space centred instead of all-positive.
        let s1 = if (h1 >> 63) & 1 == 1 { -1.0 } else { 1.0 };
        let s2 = if (h2 >> 62) & 1 == 1 { -1.0 } else { 1.0 };

        // Sublinear weighting: a token repeated 50 times is not 50x as telling.
        vector[i1] += s1;
        vector[i2] += s2 * 0.5;
    }

    // Sublinear damping, then L2 normalize so cosine is a plain dot product.
    for value in vector.iter_mut() {
        let magnitude = value.abs();
        if magnitude > 0.0 {
            *value = value.signum() * (1.0 + magnitude.ln());
        }
    }

    let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for value in vector.iter_mut() {
            *value /= norm;
        }
    }
    vector
}

/// Split text into identifier-aware tokens.
pub fn tokenize_for_embedding(text: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();

    for raw in text.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        if raw.is_empty() {
            continue;
        }
        let lower = raw.to_lowercase();
        if lower.len() > 1 {
            tokens.push(lower.clone());
        }
        // snake_case parts
        for part in lower.split('_') {
            if part.len() > 1 {
                tokens.push(part.to_string());
            }
        }
        // camelCase parts
        let mut current = String::new();
        for ch in raw.chars() {
            if ch.is_uppercase() && !current.is_empty() {
                if current.len() > 1 {
                    tokens.push(current.to_lowercase());
                }
                current.clear();
            }
            current.push(ch);
        }
        if current.len() > 1 {
            tokens.push(current.to_lowercase());
        }
    }
    tokens
}

fn fnv1a(bytes: &[u8]) -> u64 {
    fnv1a_seeded(bytes, 0xcbf2_9ce4_8422_2325)
}

fn fnv1a_seeded(bytes: &[u8], seed: u64) -> u64 {
    let mut hash = seed;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Cosine similarity of two L2-normalized vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot.clamp(-1.0, 1.0)
}

// ---------------------------------------------------------------------------
// The chunker
// ---------------------------------------------------------------------------

/// Subsystem A. Turns source files into immutable, scope-bounded chunks.
#[derive(Debug, Clone)]
pub struct AstChunker {
    /// Scopes shorter than this are folded into their parent instead of being
    /// emitted separately - a 2-line getter is not worth its own chunk.
    pub min_lines: usize,
    /// Hard ceiling. A scope longer than this is still emitted whole (never
    /// split mid-function), but flagged so the blender can rank it down.
    pub max_lines: usize,
    /// Emit container scopes (struct/impl/class) alongside their members.
    pub include_containers: bool,
}

impl Default for AstChunker {
    fn default() -> Self {
        AstChunker {
            min_lines: 1,
            max_lines: 400,
            include_containers: true,
        }
    }
}

impl AstChunker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Scan one file into chunks.
    pub fn chunk_file(&self, file: &SourceFile) -> Result<Vec<CodeChunk>, PipelineError> {
        let language = match Language::from_path(&file.path) {
            Some(lang) => lang,
            None => return Ok(Vec::new()), // unknown extension: skip, never guess
        };

        let masked = mask_source(&file.content, language);
        let masked_lines: Vec<&str> = masked.lines().collect();
        let raw_lines: Vec<&str> = file.content.lines().collect();

        if masked_lines.len() != raw_lines.len() {
            return Err(PipelineError::Parse {
                file: file.path.clone(),
                reason: format!(
                    "mask changed line count ({} -> {})",
                    raw_lines.len(),
                    masked_lines.len()
                ),
            });
        }

        // Pass 1: declarations.
        let mut declarations: Vec<Declaration> = Vec::new();
        for (index, line) in masked_lines.iter().enumerate() {
            if let Some(decl) = detect_declaration(line, language, index) {
                declarations.push(decl);
            }
        }

        // Pass 2: extents.
        let mut chunks: Vec<CodeChunk> = Vec::new();
        for decl in &declarations {
            let end_index = if language.is_brace_delimited() {
                match find_brace_scope_end(&masked_lines, decl.line_index) {
                    Some(end) => end,
                    None => continue, // declaration with no body
                }
            } else {
                find_indent_scope_end(&masked_lines, decl.line_index, decl.indent)
            };

            if end_index < decl.line_index {
                continue;
            }

            // Include immediately-preceding decorators / attributes / doc
            // comments so the chunk is self-describing.
            let mut start_index = decl.line_index;
            while start_index > 0 {
                let prev = raw_lines[start_index - 1].trim();
                let is_attached = match language {
                    Language::Rust => prev.starts_with("#[") || prev.starts_with("///") || prev.starts_with("//!"),
                    Language::Python => prev.starts_with('@'),
                    Language::JavaScript | Language::TypeScript => prev.starts_with('@') || prev.starts_with("/**") || prev.starts_with('*'),
                    Language::Go => prev.starts_with("//"),
                };
                if is_attached {
                    start_index -= 1;
                } else {
                    break;
                }
            }

            let source = raw_lines[start_index..=end_index].join("\n");
            let signature = raw_lines[decl.line_index].trim().to_string();
            let masked_body = masked_lines[decl.line_index..=end_index].join("\n");

            chunks.push(CodeChunk {
                id: String::new(), // assigned after parents are known
                file_path: file.path.clone(),
                language,
                kind: decl.kind,
                name: decl.name.clone(),
                qualified_name: decl.name.clone(),
                line_start: start_index + 1,
                line_end: end_index + 1,
                source,
                signature,
                parent: None,
                referenced_symbols: extract_referenced_symbols(&masked_body, &decl.name),
                embedding: Vec::new(),
            });
        }

        // Pass 3: nesting. The smallest strictly-enclosing chunk is the parent.
        //
        // Parent links are resolved to *indices* first and only turned into ids
        // afterwards. Ids cannot be formed inline because they are not unique
        // until deduplication has run: `struct Pool` and `impl Pool` in one file
        // both want `pool.rs::Pool`, and a colliding id silently overwrites a
        // node in the graph and the index.
        let snapshot = chunks.clone();
        let mut parent_of: Vec<Option<usize>> = vec![None; chunks.len()];
        for (index, chunk) in chunks.iter().enumerate() {
            let mut best: Option<(usize, usize)> = None; // (span, idx)
            for (other_index, other) in snapshot.iter().enumerate() {
                if other_index == index {
                    continue;
                }
                let strictly_encloses = other.line_start <= chunk.line_start
                    && other.line_end >= chunk.line_end
                    && (other.line_end - other.line_start) > (chunk.line_end - chunk.line_start);
                if strictly_encloses {
                    let span = other.line_end - other.line_start;
                    if best.map(|(b, _)| span < b).unwrap_or(true) {
                        best = Some((span, other_index));
                    }
                }
            }
            parent_of[index] = best.map(|(_, parent_index)| parent_index);
        }

        for index in 0..chunks.len() {
            if let Some(parent_index) = parent_of[index] {
                let parent_name = snapshot[parent_index].name.clone();
                let parent_kind = snapshot[parent_index].kind;
                let chunk = &mut chunks[index];
                chunk.qualified_name = format!("{}.{}", parent_name, chunk.name);
                if chunk.kind == ScopeKind::Function
                    && matches!(
                        parent_kind,
                        ScopeKind::Impl | ScopeKind::Class | ScopeKind::Trait | ScopeKind::Struct
                    )
                {
                    chunk.kind = ScopeKind::Method;
                }
            }
        }

        // Assign unique ids. Chunks are already in declaration order, so the
        // suffix a collision receives is stable across runs.
        let mut taken: HashSet<String> = HashSet::new();
        let mut assigned: Vec<String> = Vec::with_capacity(chunks.len());
        for chunk in chunks.iter() {
            let base = format!("{}::{}", chunk.file_path, chunk.qualified_name);
            let id = if taken.contains(&base) {
                // Disambiguate by kind, then by line if that still collides.
                let by_kind = format!("{}#{}", base, chunk.kind.as_str());
                if taken.contains(&by_kind) {
                    format!("{}#{}", base, chunk.line_start)
                } else {
                    by_kind
                }
            } else {
                base
            };
            taken.insert(id.clone());
            assigned.push(id);
        }

        for (index, chunk) in chunks.iter_mut().enumerate() {
            chunk.id = assigned[index].clone();
            chunk.parent = parent_of[index].map(|p| assigned[p].clone());
        }

        // Pass 4: filter, then embed.
        let mut output: Vec<CodeChunk> = Vec::new();
        for mut chunk in chunks {
            if chunk.line_count() < self.min_lines {
                continue;
            }
            if !self.include_containers
                && matches!(chunk.kind, ScopeKind::Impl | ScopeKind::Module)
            {
                continue;
            }
            chunk.embedding = compute_local_embedding(&chunk.embedding_text());
            output.push(chunk);
        }

        // Deterministic order: position in file.
        output.sort_by(|a, b| {
            a.line_start
                .cmp(&b.line_start)
                .then(a.line_end.cmp(&b.line_end))
                .then(a.id.cmp(&b.id))
        });
        Ok(output)
    }

    /// Scan many files.
    pub fn chunk_all(&self, files: &[SourceFile]) -> Result<Vec<CodeChunk>, PipelineError> {
        let mut all = Vec::new();
        for file in files {
            all.extend(self.chunk_file(file)?);
        }
        Ok(all)
    }
}

impl Filter for AstChunker {
    type Input = Vec<SourceFile>;
    type Output = ChunkIndex;

    fn name(&self) -> &'static str {
        "ast_chunker"
    }

    fn apply(
        &self,
        input: Vec<SourceFile>,
        ctx: &mut PipelineContext,
    ) -> Result<ChunkIndex, PipelineError> {
        if input.is_empty() {
            return Err(PipelineError::Empty("ast_chunker"));
        }

        let mut chunks = Vec::new();
        let mut skipped = 0u64;
        for file in &input {
            if Language::from_path(&file.path).is_none() {
                skipped += 1;
                continue;
            }
            match self.chunk_file(file) {
                Ok(mut produced) => chunks.append(&mut produced),
                // One malformed file must not sink the whole index.
                Err(err) => ctx.note(format!("skipped {}: {}", file.path, err)),
            }
        }

        ctx.set("files_indexed", input.len() as u64 - skipped);
        ctx.set("files_skipped", skipped);
        ctx.set("chunks_built", chunks.len() as u64);
        Ok(ChunkIndex::new(chunks))
    }
}

/// The searchable index produced by Subsystem A.
#[derive(Debug, Clone, Default)]
pub struct ChunkIndex {
    pub chunks: Vec<CodeChunk>,
    by_id: HashMap<String, usize>,
    by_name: HashMap<String, Vec<usize>>,
}

impl ChunkIndex {
    pub fn new(chunks: Vec<CodeChunk>) -> Self {
        let mut by_id = HashMap::new();
        let mut by_name: HashMap<String, Vec<usize>> = HashMap::new();
        for (index, chunk) in chunks.iter().enumerate() {
            by_id.insert(chunk.id.clone(), index);
            by_name
                .entry(chunk.name.to_lowercase())
                .or_default()
                .push(index);
        }
        ChunkIndex {
            chunks,
            by_id,
            by_name,
        }
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn get(&self, id: &str) -> Option<&CodeChunk> {
        self.by_id.get(id).map(|i| &self.chunks[*i])
    }

    pub fn by_name(&self, name: &str) -> Vec<&CodeChunk> {
        self.by_name
            .get(&name.to_lowercase())
            .map(|indices| indices.iter().map(|i| &self.chunks[*i]).collect())
            .unwrap_or_default()
    }

    /// Vector search. Returns `(chunk_id, cosine)` descending.
    pub fn vector_search(&self, query: &str, top_k: usize) -> Vec<(String, f32)> {
        let query_vector = compute_local_embedding(query);
        let mut scored: Vec<(String, f32)> = self
            .chunks
            .iter()
            .map(|chunk| {
                (
                    chunk.id.clone(),
                    cosine_similarity(&query_vector, &chunk.embedding),
                )
            })
            .filter(|(_, score)| *score > 0.0)
            .collect();

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0)) // deterministic tie-break
        });
        scored.truncate(top_k);
        scored
    }

    pub fn total_lines(&self) -> usize {
        self.chunks.iter().map(|c| c.line_count()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RUST_SAMPLE: &str = r#"
use std::collections::HashMap;

/// Holds pooled connections.
pub struct Pool {
    size: usize,
}

impl Pool {
    pub fn new(size: usize) -> Self {
        Pool { size }
    }

    pub fn acquire(&self) -> Option<u32> {
        let brace = "}";
        validate(brace);
        None
    }
}

fn validate(text: &str) -> bool {
    !text.is_empty()
}
"#;

    const PYTHON_SAMPLE: &str = r#"
import os


class Processor:
    """Docstring with def fake() and a } brace."""

    def __init__(self, name):
        self.name = name

    def run(self, payload):
        cleaned = sanitize(payload)
        return cleaned


def sanitize(text):
    return text.strip()
"#;

    #[test]
    fn masks_line_comments() {
        let masked = mask_source("let x = 1; // }\nlet y = 2;", Language::Rust);
        assert!(!masked.contains('}'));
        assert!(masked.contains("let x = 1;"));
        assert_eq!(masked.lines().count(), 2);
    }

    #[test]
    fn masks_string_braces() {
        let masked = mask_source(r#"let s = "}{}{";"#, Language::Rust);
        assert!(!masked.contains('}'));
        assert!(!masked.contains('{'));
    }

    #[test]
    fn masks_rust_raw_strings() {
        let src = "let s = r#\"a } brace\"#; let t = 1;";
        let masked = mask_source(src, Language::Rust);
        assert!(!masked.contains('}'));
        assert!(masked.contains("let t = 1;"));
    }

    #[test]
    fn masking_preserves_length_and_lines() {
        for (src, lang) in [
            (RUST_SAMPLE, Language::Rust),
            (PYTHON_SAMPLE, Language::Python),
        ] {
            let masked = mask_source(src, lang);
            assert_eq!(masked.len(), src.len(), "byte length must be preserved");
            assert_eq!(masked.lines().count(), src.lines().count());
        }
    }

    #[test]
    fn lifetimes_do_not_swallow_the_file() {
        let src = "fn f<'a>(x: &'a str) -> &'a str { x }\nfn g() {}";
        let masked = mask_source(src, Language::Rust);
        assert!(masked.contains("fn g()"), "masked: {}", masked);
    }

    #[test]
    fn char_literal_braces_are_masked() {
        let masked = mask_source("let c = '}';\nfn after() {}", Language::Rust);
        assert!(masked.contains("fn after()"));
        assert_eq!(masked.matches('}').count(), 1); // only the one from `{}`
    }

    #[test]
    fn python_triple_quotes_are_masked() {
        let masked = mask_source(PYTHON_SAMPLE, Language::Python);
        assert!(!masked.contains("def fake"));
        assert!(masked.contains("class Processor"));
    }

    #[test]
    fn extracts_rust_scopes() {
        let chunker = AstChunker::new();
        let chunks = chunker
            .chunk_file(&SourceFile::new("pool.rs", RUST_SAMPLE))
            .expect("chunking must succeed");

        let names: Vec<&str> = chunks.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"Pool"));
        assert!(names.contains(&"new"));
        assert!(names.contains(&"acquire"));
        assert!(names.contains(&"validate"));
    }

    #[test]
    fn a_brace_in_a_string_does_not_truncate_the_scope() {
        let chunker = AstChunker::new();
        let chunks = chunker
            .chunk_file(&SourceFile::new("pool.rs", RUST_SAMPLE))
            .unwrap();
        let acquire = chunks.iter().find(|c| c.name == "acquire").unwrap();
        assert!(
            acquire.source.contains("None"),
            "scope was cut short: {}",
            acquire.source
        );
    }

    #[test]
    fn methods_are_qualified_by_parent() {
        let chunker = AstChunker::new();
        let chunks = chunker
            .chunk_file(&SourceFile::new("pool.rs", RUST_SAMPLE))
            .unwrap();
        let acquire = chunks.iter().find(|c| c.name == "acquire").unwrap();
        assert_eq!(acquire.qualified_name, "Pool.acquire");
        assert_eq!(acquire.kind, ScopeKind::Method);
    }

    #[test]
    fn doc_comments_are_included_in_the_chunk() {
        let chunker = AstChunker::new();
        let chunks = chunker
            .chunk_file(&SourceFile::new("pool.rs", RUST_SAMPLE))
            .unwrap();
        let pool = chunks.iter().find(|c| c.name == "Pool" && c.kind == ScopeKind::Struct).unwrap();
        assert!(pool.source.contains("Holds pooled connections"));
    }

    #[test]
    fn extracts_python_scopes() {
        let chunker = AstChunker::new();
        let chunks = chunker
            .chunk_file(&SourceFile::new("proc.py", PYTHON_SAMPLE))
            .unwrap();
        let names: Vec<&str> = chunks.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"Processor"));
        assert!(names.contains(&"run"));
        assert!(names.contains(&"sanitize"));
    }

    #[test]
    fn python_scope_ends_at_dedent() {
        let chunker = AstChunker::new();
        let chunks = chunker
            .chunk_file(&SourceFile::new("proc.py", PYTHON_SAMPLE))
            .unwrap();
        let run = chunks.iter().find(|c| c.name == "run").unwrap();
        assert!(run.source.contains("return cleaned"));
        assert!(
            !run.source.contains("def sanitize"),
            "scope leaked past dedent"
        );
    }

    #[test]
    fn go_methods_carry_receiver_names() {
        let src = "package main\n\nfunc (p *Pool) Get() int {\n\treturn 1\n}\n\nfunc Helper() {}\n";
        let chunks = AstChunker::new()
            .chunk_file(&SourceFile::new("pool.go", src))
            .unwrap();
        let names: Vec<&str> = chunks.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"Get"), "got {:?}", names);
        assert!(names.contains(&"Helper"));
    }

    #[test]
    fn typescript_arrow_functions_are_found() {
        let src = "export const handler = (req: Request) => {\n  return ok(req);\n};\n";
        let chunks = AstChunker::new()
            .chunk_file(&SourceFile::new("h.ts", src))
            .unwrap();
        assert!(chunks.iter().any(|c| c.name == "handler"), "{:?}",
            chunks.iter().map(|c| &c.name).collect::<Vec<_>>());
    }

    #[test]
    fn unknown_extension_is_skipped_not_guessed() {
        let chunks = AstChunker::new()
            .chunk_file(&SourceFile::new("notes.txt", "fn looks_like_rust() {}"))
            .unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn line_coordinates_are_one_based_and_in_range() {
        let chunker = AstChunker::new();
        let total = RUST_SAMPLE.lines().count();
        for chunk in chunker
            .chunk_file(&SourceFile::new("pool.rs", RUST_SAMPLE))
            .unwrap()
        {
            assert!(chunk.line_start >= 1);
            assert!(chunk.line_end >= chunk.line_start);
            assert!(chunk.line_end <= total, "{} exceeds {}", chunk.line_end, total);
        }
    }

    #[test]
    fn source_matches_reported_coordinates() {
        let chunker = AstChunker::new();
        let lines: Vec<&str> = RUST_SAMPLE.lines().collect();
        for chunk in chunker
            .chunk_file(&SourceFile::new("pool.rs", RUST_SAMPLE))
            .unwrap()
        {
            let expected = lines[chunk.line_start - 1..chunk.line_end].join("\n");
            assert_eq!(chunk.source, expected, "coordinates disagree with source");
        }
    }

    #[test]
    fn references_are_captured_for_the_call_graph() {
        let chunker = AstChunker::new();
        let chunks = chunker
            .chunk_file(&SourceFile::new("pool.rs", RUST_SAMPLE))
            .unwrap();
        let acquire = chunks.iter().find(|c| c.name == "acquire").unwrap();
        assert!(
            acquire.referenced_symbols.iter().any(|s| s == "validate"),
            "got {:?}",
            acquire.referenced_symbols
        );
    }

    #[test]
    fn embeddings_are_normalized() {
        let v = compute_local_embedding("fn parse_config(path: &str) -> Config");
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm was {}", norm);
        assert_eq!(v.len(), EMBEDDING_DIM);
    }

    #[test]
    fn embedding_is_deterministic() {
        assert_eq!(
            compute_local_embedding("hello world"),
            compute_local_embedding("hello world")
        );
    }

    #[test]
    fn identical_text_scores_one() {
        let a = compute_local_embedding("fn charge(amount: u64)");
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn shared_identifiers_beat_unrelated_text() {
        let query = compute_local_embedding("parse config file");
        let related = compute_local_embedding("fn parse_config(path: &str) { read_file(path) }");
        let unrelated = compute_local_embedding("struct Widget { pixels: Vec<u8> }");
        assert!(
            cosine_similarity(&query, &related) > cosine_similarity(&query, &unrelated),
            "related did not outrank unrelated"
        );
    }

    #[test]
    fn empty_embedding_is_safe() {
        let v = compute_local_embedding("");
        assert_eq!(v.len(), EMBEDDING_DIM);
        assert!(v.iter().all(|x| *x == 0.0));
        assert_eq!(cosine_similarity(&v, &v), 0.0);
    }

    #[test]
    fn vector_search_ranks_the_right_chunk_first() {
        let chunker = AstChunker::new();
        let chunks = chunker
            .chunk_all(&[
                SourceFile::new("pool.rs", RUST_SAMPLE),
                SourceFile::new("proc.py", PYTHON_SAMPLE),
            ])
            .unwrap();
        let index = ChunkIndex::new(chunks);
        let hits = index.vector_search("sanitize payload text", 5);
        assert!(!hits.is_empty());
        assert!(
            hits[0].0.contains("sanitize"),
            "expected sanitize first, got {:?}",
            hits
        );
    }

    #[test]
    fn chunks_of_a_scope_never_split_it() {
        // Every emitted chunk must be a complete brace-balanced scope.
        let chunker = AstChunker::new();
        for chunk in chunker
            .chunk_file(&SourceFile::new("pool.rs", RUST_SAMPLE))
            .unwrap()
        {
            if !chunk.source.contains('{') {
                continue;
            }
            let masked = mask_source(&chunk.source, Language::Rust);
            let opens = masked.matches('{').count();
            let closes = masked.matches('}').count();
            assert_eq!(opens, closes, "unbalanced chunk: {}", chunk.id);
        }
    }

    #[test]
    fn filter_reports_counters() {
        let mut ctx = PipelineContext::new();
        let index = AstChunker::new()
            .run(
                vec![
                    SourceFile::new("pool.rs", RUST_SAMPLE),
                    SourceFile::new("readme.md", "# not code"),
                ],
                &mut ctx,
            )
            .unwrap();
        assert!(!index.is_empty());
        assert_eq!(ctx.counter("files_skipped"), 1);
        assert_eq!(ctx.counter("chunks_built"), index.len() as u64);
    }

    #[test]
    fn empty_input_is_an_error() {
        let mut ctx = PipelineContext::new();
        assert!(AstChunker::new().run(Vec::new(), &mut ctx).is_err());
    }

    #[test]
    fn index_lookup_by_name_is_case_insensitive() {
        let chunks = AstChunker::new()
            .chunk_file(&SourceFile::new("pool.rs", RUST_SAMPLE))
            .unwrap();
        let index = ChunkIndex::new(chunks);
        assert!(!index.by_name("VALIDATE").is_empty());
    }
}
