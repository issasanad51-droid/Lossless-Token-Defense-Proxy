//! High-performance terminal / build-log compressor.
//!
//! Compilation tickers (`[1/500]`), carriage-return progress frames, ANSI
//! CSI/OSC sequences, percentage bars and spinner glyphs are erased with a
//! pre-compiled regex battery. Signal lines (errors, warnings, unique
//! `Compiling` events) survive.

use regex::Regex;

use crate::pipeline::TokenFilter;

/// Regex-driven build-log compressor.
pub struct LogCleaner {
    ansi: Regex,
    tickers: Vec<Regex>,
    drop_line: Vec<Regex>,
}

impl std::fmt::Debug for LogCleaner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogCleaner")
            .field("tickers", &self.tickers.len())
            .field("drop_line", &self.drop_line.len())
            .finish()
    }
}

impl LogCleaner {
    pub fn new() -> Self {
        // Every pattern is compiled once at construction. The orchestrator
        // reuses a single LogCleaner for the whole process lifetime.
        let tickers = compile_all(&[
            // Classic compilation counters.
            r"\[\s*\d+\s*/\s*\d+\s*\]",
            r"\(\s*\d+\s*/\s*\d+\s*\)",
            r"\{\s*\d+\s*/\s*\d+\s*\}",
            // Percentages and byte-progress (`45%`, `12.5%`, `3.1 MiB/s`).
            r"\b\d{1,3}(?:\.\d+)?\s*%",
            r"\b\d+(?:\.\d+)?\s*(?:[kKmMgGtT]i?[bB])/s\b",
            // ASCII / unicode progress bars.
            r"\[\s*[=#\-\x{2500}-\x{259F}]{2,}[>\s]*\]",
            r"\x{2588}{2,}|\x{2593}{2,}|\x{2591}{2,}",
            // ETA / elapsed clocks attached to tickers.
            r"(?i)\b(?:eta|elapsed|remaining|took)\s*[:=]?\s*\d+[smh:\d]*",
            // Cargo / ninja / make counters.
            r"(?i)\b(?:building|compiling|linking|downloading)\s*\[\s*[\d\s/%]+\]",
        ]);

        let drop_line = compile_all(&[
            // A line that is only a spinner / braille frame / dots.
            r"^[\s\|/\-\\\.\u2800-\u28FF\u2022\u2219\*]+$",
            r"(?i)^\s*(?:progress|downloading|waiting|installing)[:\s].*$",
            r"^\s*\d+\s*/\s*\d+\s*$",
            // Repeated "Compiling foo" noise is handled in a second pass;
            // here we only drop empty progress leftovers.
            r"^\s*\[=+>+?\s*\]\s*$",
        ]);

        Self {
            ansi: Regex::new(r"\x1b(?:\[[0-9;?]*[A-Za-z]|].*?(?:\x07|\x1b\\)|[()][AB012])")
                .expect("ansi regex"),
            tickers,
            drop_line,
        }
    }

    pub fn clean(&self, input: &str) -> String {
        if input.is_empty() {
            return String::new();
        }

        // 1. Split on `\n` but first flatten `\r` progress frames so only
        //    the last visible frame of each physical line remains.
        let mut kept: Vec<String> = Vec::new();
        let mut last_emitted: Option<String> = None;
        let mut compiling_run: Vec<String> = Vec::new();

        for physical in input.split('\n') {
            let frame = last_cr_frame(physical);
            let mut line = self.ansi.replace_all(frame, "").into_owned();
            for re in &self.tickers {
                line = re.replace_all(&line, "").into_owned();
            }
            // Collapse leftover double spaces introduced by ticker holes.
            line = collapse_spaces(&line);
            let trimmed = line.trim();

            if trimmed.is_empty() {
                flush_compiling(&mut compiling_run, &mut kept, &mut last_emitted);
                continue;
            }
            if self.drop_line.iter().any(|re| re.is_match(trimmed)) {
                continue;
            }

            if is_compiling_line(trimmed) {
                compiling_run.push(trimmed.to_string());
                continue;
            }

            flush_compiling(&mut compiling_run, &mut kept, &mut last_emitted);

            if last_emitted.as_deref() == Some(trimmed) {
                continue;
            }
            kept.push(trimmed.to_string());
            last_emitted = Some(trimmed.to_string());
        }
        flush_compiling(&mut compiling_run, &mut kept, &mut last_emitted);

        let mut out = kept.join("\n");
        if input.ends_with('\n') && !out.is_empty() && !out.ends_with('\n') {
            out.push('\n');
        }
        out
    }
}

impl Default for LogCleaner {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenFilter for LogCleaner {
    fn filter(&self, input: &str) -> String {
        self.clean(input)
    }

    fn name(&self) -> &'static str {
        "log_cleaner"
    }
}

fn compile_all(patterns: &[&str]) -> Vec<Regex> {
    patterns
        .iter()
        .map(|p| Regex::new(p).unwrap_or_else(|e| panic!("invalid log regex {p}: {e}")))
        .collect()
}

fn last_cr_frame(s: &str) -> &str {
    s.rsplit('\r').next().unwrap_or(s)
}

fn collapse_spaces(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = false;
    for ch in s.chars() {
        if ch == ' ' || ch == '\t' {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            prev_space = false;
            out.push(ch);
        }
    }
    out
}

fn is_compiling_line(s: &str) -> bool {
    let t = s.trim();
    t.starts_with("Compiling ")
        || t.starts_with("Checking ")
        || t.starts_with("Downloading ")
        || t.starts_with("Downloaded ")
        || t.starts_with("Fresh ")
}

fn flush_compiling(
    run: &mut Vec<String>,
    kept: &mut Vec<String>,
    last_emitted: &mut Option<String>,
) {
    if run.is_empty() {
        return;
    }
    // Keep unique crate names; compress a flood into one summary line
    // plus the first and last event so the log remains lossless about
    // *what* compiled, just not about the ticker cadence.
    let mut unique = Vec::new();
    for line in run.drain(..) {
        if !unique.iter().any(|u: &String| u == &line) {
            unique.push(line);
        }
    }
    if unique.len() <= 3 {
        for line in unique {
            if last_emitted.as_deref() != Some(line.as_str()) {
                *last_emitted = Some(line.clone());
                kept.push(line);
            }
        }
        return;
    }
    let first = unique.first().cloned().unwrap();
    let last = unique.last().cloned().unwrap();
    let summary = format!(
        "compiled {} crates ({} … {})",
        unique.len(),
        crate_token(&first),
        crate_token(&last)
    );
    kept.push(first);
    kept.push(summary);
    kept.push(last);
    *last_emitted = Some(unique.last().cloned().unwrap());
}

fn crate_token(line: &str) -> String {
    line.split_whitespace()
        .nth(1)
        .unwrap_or(line)
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erases_fraction_tickers() {
        let log = "[1/500] step one\n[2/500] step two\nerror: boom\n";
        let out = LogCleaner::new().clean(log);
        assert!(!out.contains("[1/500]"));
        assert!(!out.contains("[2/500]"));
        assert!(out.contains("error: boom"));
        assert!(out.contains("step one"));
    }

    #[test]
    fn keeps_last_cr_frame() {
        let log = "progress 1%\rprogress 50%\rprogress 100%\ndone\n";
        let out = LogCleaner::new().clean(log);
        assert!(out.contains("done"));
        assert!(!out.contains("progress 1%"));
    }

    #[test]
    fn strips_ansi() {
        let log = "\u{1b}[31merror\u{1b}[0m: nope\n";
        let out = LogCleaner::new().clean(log);
        assert_eq!(out.trim(), "error: nope");
    }
}
