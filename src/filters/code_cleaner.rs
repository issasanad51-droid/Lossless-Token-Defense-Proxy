//! Lossless line-by-line comment and whitespace stripper.
//!
//! The scanner is a single-pass UTF-8 state machine. It never rewrites
//! identifiers, literals, or indentation that carries meaning. Comments are
//! dropped; functional leading whitespace on surviving code lines is copied
//! byte-for-byte.

use crate::pipeline::TokenFilter;

/// Strip comments / redundant whitespace without touching program meaning.
#[derive(Debug, Clone)]
pub struct CodeCleaner {
    /// Keep a leading `#!` interpreter line (scripts).
    pub preserve_shebang: bool,
    /// Collapse runs of blank lines down to a single blank line.
    pub collapse_blank_lines: bool,
}

impl CodeCleaner {
    pub fn new() -> Self {
        Self {
            preserve_shebang: true,
            collapse_blank_lines: true,
        }
    }

    /// Core transform. Public so the test runner can call it without the trait.
    pub fn clean(&self, input: &str) -> String {
        if input.is_empty() {
            return String::new();
        }

        let kept = mark_kept_bytes(input);
        let mut raw = String::with_capacity(input.len());
        for (i, ch) in input.char_indices() {
            let width = ch.len_utf8();
            let keep = (0..width).all(|k| kept[i + k]);
            if keep {
                raw.push(ch);
            }
        }

        self.postprocess_lines(&raw)
    }

    fn postprocess_lines(&self, raw: &str) -> String {
        let mut out = String::with_capacity(raw.len());
        let mut blank_run = 0usize;
        let mut first = true;

        for (idx, line) in raw.split_inclusive('\n').enumerate() {
            let had_nl = line.ends_with('\n');
            let body = if had_nl { &line[..line.len() - 1] } else { line };
            // Preserve CR if someone fed us CRLF; strip it then re-emit `\n`.
            let body = body.strip_suffix('\r').unwrap_or(body);
            let trimmed_right = rtrim_ws(body);

            if self.preserve_shebang && idx == 0 && trimmed_right.starts_with("#!") {
                out.push_str(trimmed_right);
                if had_nl {
                    out.push('\n');
                }
                first = false;
                blank_run = 0;
                continue;
            }

            if trimmed_right.is_empty() {
                blank_run += 1;
                if !self.collapse_blank_lines {
                    if !first {
                        out.push('\n');
                    }
                    first = false;
                }
                continue;
            }

            if self.collapse_blank_lines && blank_run > 0 && !first {
                out.push('\n');
            }
            blank_run = 0;
            out.push_str(trimmed_right);
            if had_nl {
                out.push('\n');
            }
            first = false;
        }

        out
    }
}

impl Default for CodeCleaner {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenFilter for CodeCleaner {
    fn filter(&self, input: &str) -> String {
        self.clean(input)
    }

    fn name(&self) -> &'static str {
        "code_cleaner"
    }
}

fn rtrim_ws(s: &str) -> &str {
    s.trim_end_matches(|c: char| c == ' ' || c == '\t' || c == '\u{000c}')
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Code,
    LineComment,
    BlockComment { saw_star: bool },
    String {
        delim: u8,
        escaped: bool,
        triple: bool,
        triple_seen: u8,
    },
    RawString {
        hashes: u8,
        closing: bool,
        seen: u8,
    },
}

/// Mark every source byte as kept (`true`) or comment (`false`).
/// Newlines inside comments stay kept so line structure of surrounding code
/// cannot collapse together.
fn mark_kept_bytes(src: &str) -> Vec<bool> {
    let b = src.as_bytes();
    let n = b.len();
    let mut keep = vec![true; n];
    let mut i = 0usize;
    let mut state = State::Code;

    while i < n {
        match state {
            State::Code => {
                // Shebang / rust attributes / python comments.
                if b[i] == b'#' {
                    if is_rust_raw_prefix(b, i) {
                        let hashes = count_hashes(b, i + 1);
                        let after = i + 1 + hashes;
                        if after < n && b[after] == b'"' {
                            state = State::RawString {
                                hashes: hashes as u8,
                                closing: false,
                                seen: 0,
                            };
                            i = after + 1;
                            continue;
                        }
                    }
                    if i + 1 < n && (b[i + 1] == b'[' || b[i + 1] == b'!') {
                        // `#[attr]` or `#![attr]` or `#!` shebang — keep.
                        i += 1;
                        continue;
                    }
                    // Python / shell / ruby line comment.
                    mark_line_comment(&mut keep, b, i);
                    state = State::LineComment;
                    i += 1;
                    continue;
                }

                if b[i] == b'/' && i + 1 < n && b[i + 1] == b'/' {
                    mark_line_comment(&mut keep, b, i);
                    state = State::LineComment;
                    i += 2;
                    continue;
                }

                if b[i] == b'/' && i + 1 < n && b[i + 1] == b'*' {
                    keep[i] = false;
                    keep[i + 1] = false;
                    state = State::BlockComment { saw_star: false };
                    i += 2;
                    continue;
                }

                if b[i] == b'"' || b[i] == b'\'' || b[i] == b'`' {
                    if b[i] == b'\'' && is_rust_lifetime(b, i) {
                        i += 1;
                        continue;
                    }
                    let triple = is_triple(b, i);
                    if triple {
                        state = State::String {
                            delim: b[i],
                            escaped: false,
                            triple: true,
                            triple_seen: 0,
                        };
                        i += 3;
                    } else {
                        state = State::String {
                            delim: b[i],
                            escaped: false,
                            triple: false,
                            triple_seen: 0,
                        };
                        i += 1;
                    }
                    continue;
                }

                // `r"..."`, `r#"..."#`, `br#"..."#`, `cr#"..."#`.
                if (b[i] == b'r'
                    || ((b[i] == b'b' || b[i] == b'c') && i + 1 < n && b[i + 1] == b'r'))
                    && starts_raw_string(b, i)
                {
                    let (hashes, quote_at) = raw_string_header(b, i);
                    state = State::RawString {
                        hashes: hashes as u8,
                        closing: false,
                        seen: 0,
                    };
                    i = quote_at + 1;
                    continue;
                }

                i += 1;
            }
            State::LineComment => {
                if b[i] == b'\n' {
                    // Keep the newline; it belongs to the file, not the comment.
                    state = State::Code;
                } else {
                    keep[i] = false;
                }
                i += 1;
            }
            State::BlockComment { saw_star } => {
                if b[i] == b'\n' {
                    keep[i] = true;
                    state = State::BlockComment { saw_star: false };
                    i += 1;
                    continue;
                }
                keep[i] = false;
                if saw_star && b[i] == b'/' {
                    state = State::Code;
                    i += 1;
                    continue;
                }
                state = State::BlockComment {
                    saw_star: b[i] == b'*',
                };
                i += 1;
            }
            State::String {
                delim,
                escaped,
                triple,
                triple_seen,
            } => {
                if escaped {
                    state = State::String {
                        delim,
                        escaped: false,
                        triple,
                        triple_seen: 0,
                    };
                    i += 1;
                    continue;
                }
                if !triple && b[i] == b'\\' {
                    state = State::String {
                        delim,
                        escaped: true,
                        triple,
                        triple_seen: 0,
                    };
                    i += 1;
                    continue;
                }
                if triple {
                    if b[i] == delim {
                        let next = triple_seen + 1;
                        if next == 3 {
                            state = State::Code;
                            i += 1;
                            continue;
                        }
                        state = State::String {
                            delim,
                            escaped: false,
                            triple: true,
                            triple_seen: next,
                        };
                    } else {
                        state = State::String {
                            delim,
                            escaped: false,
                            triple: true,
                            triple_seen: 0,
                        };
                    }
                    i += 1;
                    continue;
                }
                if b[i] == delim {
                    state = State::Code;
                }
                i += 1;
            }
            State::RawString {
                hashes,
                closing,
                seen,
            } => {
                if !closing {
                    if b[i] == b'"' {
                        state = State::RawString {
                            hashes,
                            closing: true,
                            seen: 0,
                        };
                    }
                    i += 1;
                    continue;
                }
                if hashes == 0 {
                    state = State::Code;
                    continue;
                }
                if b[i] == b'#' {
                    let next = seen + 1;
                    if next == hashes {
                        state = State::Code;
                        i += 1;
                        continue;
                    }
                    state = State::RawString {
                        hashes,
                        closing: true,
                        seen: next,
                    };
                    i += 1;
                } else if b[i] == b'"' {
                    state = State::RawString {
                        hashes,
                        closing: true,
                        seen: 0,
                    };
                    i += 1;
                } else {
                    state = State::RawString {
                        hashes,
                        closing: false,
                        seen: 0,
                    };
                    i += 1;
                }
            }
        }
    }

    keep
}

fn mark_line_comment(keep: &mut [bool], b: &[u8], start: usize) {
    let mut j = start;
    while j < b.len() && b[j] != b'\n' {
        keep[j] = false;
        j += 1;
    }
}

fn is_triple(b: &[u8], i: usize) -> bool {
    i + 2 < b.len() && b[i] == b[i + 1] && b[i] == b[i + 2]
}

fn is_rust_lifetime(b: &[u8], i: usize) -> bool {
    // `'a`, `'static`, `'_` — but not `'x'` (char) and not `'hello'` (py string).
    if i + 1 >= b.len() {
        return false;
    }
    let n = b[i + 1];
    if !(n == b'_' || n.is_ascii_alphabetic()) {
        return false;
    }
    // `'x'` is a char literal.
    if i + 2 < b.len() && b[i + 2] == b'\'' {
        return false;
    }
    // `'\\'` style chars are handled by the `\` branch in the caller (next
    // byte is not alphabetic). Python strings like `'hello'` would look like
    // a lifetime under this rule — we only treat it as a lifetime when the
    // following ident is then a non-string token (`>`, `,`, space, `:`, `)`).
    let mut k = i + 2;
    while k < b.len() && (b[k].is_ascii_alphanumeric() || b[k] == b'_') {
        k += 1;
    }
    if k >= b.len() {
        return true;
    }
    matches!(
        b[k],
        b'>' | b',' | b')' | b';' | b'|' | b'&' | b'+' | b'=' | b' ' | b'\t' | b'\n' | b'\r'
    )
}

fn is_rust_raw_prefix(b: &[u8], hash_at: usize) -> bool {
    // Called when we see `#`. True if this `#` is part of `r##"`.
    if hash_at == 0 {
        return false;
    }
    let mut k = hash_at;
    while k > 0 && b[k - 1] == b'#' {
        k -= 1;
    }
    if k == 0 {
        return false;
    }
    if b[k - 1] == b'r' {
        return true;
    }
    if k >= 2 && b[k - 1] == b'r' && (b[k - 2] == b'b' || b[k - 2] == b'c') {
        return true;
    }
    false
}

fn count_hashes(b: &[u8], mut i: usize) -> usize {
    let start = i;
    while i < b.len() && b[i] == b'#' {
        i += 1;
    }
    i - start
}

fn starts_raw_string(b: &[u8], i: usize) -> bool {
    let r = if b[i] == b'r' {
        i
    } else if (b[i] == b'b' || b[i] == b'c') && i + 1 < b.len() && b[i + 1] == b'r' {
        i + 1
    } else {
        return false;
    };
    let mut k = r + 1;
    while k < b.len() && b[k] == b'#' {
        k += 1;
    }
    k < b.len() && b[k] == b'"'
}

fn raw_string_header(b: &[u8], i: usize) -> (usize, usize) {
    let r = if b[i] == b'r' { i } else { i + 1 };
    let mut k = r + 1;
    let mut hashes = 0usize;
    while k < b.len() && b[k] == b'#' {
        hashes += 1;
        k += 1;
    }
    (hashes, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_rust_line_comment_keeps_url() {
        let out = CodeCleaner::new().clean("let u = \"http://x.com\"; // trail\n");
        assert!(out.contains("http://x.com"));
        assert!(!out.contains("trail"));
    }

    #[test]
    fn keeps_hash_inside_python_string() {
        let out = CodeCleaner::new().clean("x = \"hello # world\"  # bye\n");
        assert!(out.contains("hello # world"));
        assert!(!out.contains("bye"));
    }

    #[test]
    fn preserves_indentation() {
        let out = CodeCleaner::new().clean("def f():\n    return 1  # x\n");
        assert!(out.contains("    return 1"));
    }

    #[test]
    fn preserves_shebang() {
        let out = CodeCleaner::new().clean("#!/usr/bin/env python3\n# comment\nx = 1\n");
        assert!(out.starts_with("#!/usr/bin/env python3"));
        assert!(!out.contains("comment"));
    }

    #[test]
    fn keeps_rust_attributes() {
        let out = CodeCleaner::new().clean("#[derive(Debug)]\nstruct Foo;\n");
        assert!(out.contains("#[derive(Debug)]"));
    }

    #[test]
    fn strips_block_comments_without_gluing_lines() {
        let src = "let a = 1;\n/* comment\nstill */\nlet b = 2;\n";
        let out = CodeCleaner::new().clean(src);
        assert!(out.contains("let a = 1;"));
        assert!(out.contains("let b = 2;"));
        assert!(!out.contains("comment"));
    }

    #[test]
    fn keeps_rust_lifetimes() {
        let src = "fn foo<'a>(x: &'a str) { x }\n";
        let out = CodeCleaner::new().clean(src);
        assert!(out.contains("fn foo<'a>(x: &'a str)"));
    }
}
