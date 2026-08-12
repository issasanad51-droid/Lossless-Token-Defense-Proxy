//! Lossless compressors: source comments and terminal progress noise.

use once_cell::sync::Lazy;
use regex::Regex;

/// Tuning knobs for [`lossless_code_compressor`].
#[derive(Debug, Clone)]
pub struct CodeOptions {
    pub max_consecutive_blank_lines: usize,
    pub strip_trailing_comments: bool,
    pub keep_semantic_comments: bool,
}

impl Default for CodeOptions {
    fn default() -> Self {
        Self {
            max_consecutive_blank_lines: 1,
            strip_trailing_comments: true,
            keep_semantic_comments: true,
        }
    }
}

static SEMANTIC_COMMENT: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?ix)^\#\s*(?:
              !
            | -\*-
            | (?:type|mypy|pyright)\b
            | noqa\b
            | pragma\b
            | pylint\b
            | flake8\b
            | ruff\b
            | fmt\s*:\s*(?:on|off|skip)
            | yapf\s*:\s*(?:disable|enable)
            | isort\s*:\s*
            | nosec\b
            | coding[:=]
        )",
    )
    .expect("valid semantic-comment regex")
});

/// Scan one line, tracking string state. Returns the byte index of a real
/// comment start (if any) and the triple-quote delimiter still open at EOL.
fn scan_line(line: &str, in_triple: Option<&str>) -> (Option<usize>, Option<&'static str>) {
    let bytes: Vec<char> = line.chars().collect();
    let n = bytes.len();
    let mut i = 0usize;
    // Map char index -> byte index so we can return a byte offset for slicing.
    let byte_at = |ci: usize| -> usize { line.char_indices().nth(ci).map(|(b, _)| b).unwrap_or(line.len()) };

    let mut open: Option<&'static str> = match in_triple {
        Some("\"\"\"") => Some("\"\"\""),
        Some("'''") => Some("'''"),
        _ => None,
    };

    if let Some(delim) = open {
        let d: Vec<char> = delim.chars().collect();
        while i < n {
            if bytes[i] == '\\' {
                i += 2;
                continue;
            }
            if i + d.len() <= n && bytes[i..i + d.len()] == d[..] {
                i += d.len();
                open = None;
                break;
            }
            i += 1;
        }
        if open.is_some() {
            return (None, open);
        }
    }

    while i < n {
        let ch = bytes[i];

        if ch == '#' {
            return (Some(byte_at(i)), None);
        }

        if ch == '\'' || ch == '"' {
            let triple: &'static str = if ch == '"' { "\"\"\"" } else { "'''" };
            let t: Vec<char> = triple.chars().collect();
            if i + 3 <= n && bytes[i..i + 3] == t[..] {
                i += 3;
                let mut closed = false;
                while i < n {
                    if bytes[i] == '\\' {
                        i += 2;
                        continue;
                    }
                    if i + 3 <= n && bytes[i..i + 3] == t[..] {
                        i += 3;
                        closed = true;
                        break;
                    }
                    i += 1;
                }
                if !closed {
                    return (None, Some(triple));
                }
                continue;
            }

            let quote = ch;
            i += 1;
            while i < n {
                if bytes[i] == '\\' {
                    i += 2;
                    continue;
                }
                if bytes[i] == quote {
                    i += 1;
                    break;
                }
                i += 1;
            }
            continue;
        }

        i += 1;
    }

    (None, None)
}

/// Remove human comments and collapse blank-line runs without touching
/// indentation, string contents or any executable token.
pub fn lossless_code_compressor(raw_code: &str, opts: &CodeOptions) -> String {
    if raw_code.is_empty() {
        return String::new();
    }

    let mut out: Vec<String> = Vec::new();
    let mut in_triple: Option<&'static str> = None;
    let mut blank_run = 0usize;

    for (idx, line) in raw_code.lines().enumerate() {
        let was_in_string = in_triple.is_some();
        let (comment_start, next_triple) = scan_line(line, in_triple);
        in_triple = next_triple;

        if was_in_string {
            out.push(line.to_string());
            blank_run = 0;
            continue;
        }

        let mut current: String;

        if let Some(cs) = comment_start {
            let comment_text = &line[cs..];
            let before = line[..cs].trim();
            let whole_line = before.is_empty();

            let keep = opts.keep_semantic_comments
                && (SEMANTIC_COMMENT.is_match(comment_text.trim())
                    || (idx == 0 && comment_text.starts_with("#!")));

            if keep {
                out.push(line.trim_end().to_string());
                blank_run = 0;
                continue;
            }
            if whole_line {
                continue;
            }
            current = if opts.strip_trailing_comments {
                line[..cs].trim_end().to_string()
            } else {
                line.trim_end().to_string()
            };
        } else {
            current = line.trim_end().to_string();
        }

        if current.trim().is_empty() {
            blank_run += 1;
            if blank_run > opts.max_consecutive_blank_lines {
                continue;
            }
            current = String::new();
            out.push(current);
            continue;
        }

        blank_run = 0;
        out.push(current);
    }

    while out.first().map_or(false, |l| l.trim().is_empty()) {
        out.remove(0);
    }
    while out.last().map_or(false, |l| l.trim().is_empty()) {
        out.pop();
    }

    out.join("\n")
}

// ---------------------------------------------------------------------------
// Terminal cleaning
// ---------------------------------------------------------------------------

static SIGNAL: Lazy<Vec<Regex>> = Lazy::new(|| {
    [
        r"(?i)\b(?:error|errors)\b",
        r"(?i)\b(?:fatal|panic|abort(?:ed)?|segfault|core dumped)\b",
        r"(?i)\b(?:fail|failed|failure|failing)\b",
        r"(?i)\b(?:exception|traceback|stack ?trace|assertion)\b",
        r"(?i)\b(?:undefined reference|cannot find|not found|no such file)\b",
        r"(?i)\b(?:exit(?:ed)? (?:code|status)|exit code)\b",
        r"(?i)\b(?:succeeded|success|successful|completed successfully)\b",
        r"(?i)\b(?:build (?:finished|complete|succeeded|failed))\b",
        r"(?i)\b(?:finished|done) in\b",
        r"^\s*(?:E|ERROR|FAIL|FAILED|CRITICAL)\b",
        r"(?i)\bwarning\b.*\b(?:treated as error|-Werror)\b",
        r#"^\s*(?:at |File "|\s+\^+\s*$)"#,
    ]
    .iter()
    .map(|p| Regex::new(p).expect("valid signal regex"))
    .collect()
});

static NOISE: Lazy<Vec<Regex>> = Lazy::new(|| {
    [
        r"^\s*\[\s*\d+\s*/\s*\d+\s*\]",
        r"^\s*\(\s*\d+\s*/\s*\d+\s*\)",
        r"^\s*\d{1,3}\s*%",
        r"(?i)\b\d{1,3}(?:\.\d+)?\s*%\s*(?:complete|completed|done|finished)?\b",
        r"(?i)^\s*(?:Compiling|Building|Downloading|Fetching|Extracting|Unpacking|Installing|Resolving|Linking|Indexing|Uploading|Pulling|Cloning)\b",
        r"\b(?:ETA|eta)\s+\d",
        r"\b\d+(?:\.\d+)?\s*(?:[KMG]i?B)\s*/\s*\d+(?:\.\d+)?\s*(?:[KMG]i?B)",
        r"\b\d+(?:\.\d+)?\s*(?:[KMG]i?B|B)/s\b",
        r"^[\s\|\-\\/\*\.=#>█░▒▓►⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]+$",
        r"^\s*(?:\[=*>?\s*\]|\[#*\s*\]|\[\.*\s*\])\s*$",
        r"(?i)^\s*(?:Progress|Status):",
        r"(?i)^\s*(?:Receiving|Counting|Compressing|Delta) objects:",
        r"(?i)^\s*remote:\s*(?:Counting|Compressing|Enumerating|Total)\b",
        r"^\s*(?:Reading (?:package|state)|Get:\d+|Hit:\d+|Ign:\d+)\b",
        r"^\s*(?:npm|yarn|pnpm)\s+(?:WARN\s+)?(?:idealTree|timing|sill|http fetch)\b",
        r"^\s*\.{3,}\s*$",
    ]
    .iter()
    .map(|p| Regex::new(p).expect("valid noise regex"))
    .collect()
});

static ANSI: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07").expect("valid ansi regex"));

fn is_signal(line: &str) -> bool {
    SIGNAL.iter().any(|r| r.is_match(line))
}

fn is_noise(line: &str) -> bool {
    NOISE.iter().any(|r| r.is_match(line))
}

/// Strip build tickers and download loops; keep failures and completions.
pub fn lossless_terminal_cleaner(raw_logs: &str, annotate_drops: bool, collapse_duplicates: bool) -> String {
    if raw_logs.is_empty() {
        return String::new();
    }

    let mut out: Vec<String> = Vec::new();
    let mut dropped = 0usize;
    let mut prev: Option<String> = None;
    let mut prev_count = 0usize;

    macro_rules! flush_dupes {
        () => {
            if let Some(p) = prev.take() {
                if prev_count > 1 {
                    out.push(format!("{}  (x{})", p, prev_count));
                } else {
                    out.push(p);
                }
                prev_count = 0;
            }
        };
    }

    for raw in raw_logs.lines() {
        // Keep only the final frame of a \r-redrawn line.
        let last_frame = raw.rsplit('\r').next().unwrap_or(raw);
        let cleaned = ANSI.replace_all(last_frame, "");
        let line = cleaned.trim_end();

        if line.trim().is_empty() {
            continue;
        }

        if is_signal(line) || !is_noise(line) {
            if dropped > 0 && annotate_drops {
                flush_dupes!();
                out.push(format!("... {} progress lines omitted ...", dropped));
            }
            dropped = 0;

            if collapse_duplicates && prev.as_deref() == Some(line) {
                prev_count += 1;
            } else {
                flush_dupes!();
                prev = Some(line.to_string());
                prev_count = 1;
            }
            continue;
        }

        flush_dupes!();
        dropped += 1;
    }

    flush_dupes!();
    if dropped > 0 && annotate_drops {
        out.push(format!("... {} progress lines omitted ...", dropped));
    }

    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_inside_string_survives() {
        let src = "note = \"refund # not a comment\"  # real comment\n";
        let out = lossless_code_compressor(src, &CodeOptions::default());
        assert_eq!(out, "note = \"refund # not a comment\"");
    }

    #[test]
    fn whole_line_comments_removed() {
        let src = "# chatter\nx = 1\n";
        assert_eq!(lossless_code_compressor(src, &CodeOptions::default()), "x = 1");
    }

    #[test]
    fn semantic_comments_kept() {
        let src = "#!/usr/bin/env python3\n# chatter\nimport os  # noqa\n";
        let out = lossless_code_compressor(src, &CodeOptions::default());
        assert!(out.contains("#!/usr/bin/env python3"));
        assert!(out.contains("noqa"));
        assert!(!out.contains("chatter"));
    }

    #[test]
    fn indentation_preserved() {
        let src = "if x:\n    if y:\n        deep()  # go\n";
        let out = lossless_code_compressor(src, &CodeOptions::default());
        assert!(out.contains("        deep()"));
    }

    #[test]
    fn blank_runs_collapse() {
        let src = "a = 1\n\n\n\n\nb = 2\n";
        assert_eq!(lossless_code_compressor(src, &CodeOptions::default()), "a = 1\n\nb = 2");
    }

    #[test]
    fn triple_quoted_block_untouched() {
        let src = "x = \"\"\"a\n\n\n\nb\"\"\"\n";
        let out = lossless_code_compressor(src, &CodeOptions::default());
        assert!(out.contains("a\n\n\n\nb"));
    }

    #[test]
    fn errors_survive_cleaning() {
        let logs = "[1/250] Compiling a.c\n 45% completed\nsrc/pool.c:88: error: bad member\nBuild failed in 48.2s\n";
        let out = lossless_terminal_cleaner(logs, true, true);
        assert!(out.contains("error: bad member"));
        assert!(out.contains("Build failed"));
        assert!(!out.contains("Compiling a.c"));
    }

    #[test]
    fn signal_beats_noise() {
        let logs = "[42/250] error: on fire\n[43/250] Compiling ok.c\n";
        let out = lossless_terminal_cleaner(logs, true, true);
        assert!(out.contains("on fire"));
        assert!(!out.contains("ok.c"));
    }

    #[test]
    fn ansi_stripped() {
        let out = lossless_terminal_cleaner("\x1b[31mfatal: broken\x1b[0m\n", true, true);
        assert_eq!(out, "fatal: broken");
    }

    #[test]
    fn duplicates_collapse() {
        let out = lossless_terminal_cleaner("same error\nsame error\nsame error\n", true, true);
        assert_eq!(out, "same error  (x3)");
    }

    #[test]
    fn empty_inputs() {
        assert_eq!(lossless_code_compressor("", &CodeOptions::default()), "");
        assert_eq!(lossless_terminal_cleaner("", true, true), "");
    }
}
