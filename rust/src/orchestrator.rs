//! The middleware pipeline: compress, guard, measure.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::compressors::{lossless_code_compressor, lossless_terminal_cleaner, CodeOptions};
use crate::data_converter::json_to_minimal_yaml;
use crate::tokenizer::TokenCounter;

/// Strict output contract for the downstream model.
pub const CAVEMAN_GUARDRAIL: &str = "\
OUTPUT PROTOCOL (STRICT - violations make the response unusable):
1 Reply ONLY with unified diffs or exact replacement blocks. No prose around them.
2 BANNED openers: greetings, \"Certainly\", \"Great question\", \"I'd be happy to\",
  restating the task, summarizing what you are about to do.
3 BANNED closers: summaries, \"Let me know if\", next-step offers, congratulations.
4 No explanation unless a line starts with WHY: - max 1 such line, max 15 words.
5 Every hunk needs a file path and line anchor. Never reprint an unchanged file.
6 Never reprint unchanged functions/imports to give context. Diff only.
7 Uncertain? Emit ASK: <one line>. Do not guess and do not hedge in prose.
8 No markdown headers, no bullet recaps, no emoji, no apologies.
9 Code comments in your patch: only where logic is non-obvious. No narration.
10 Caveman register: terse, imperative, zero filler words.";

#[derive(Debug, Clone, Copy)]
pub struct SectionStat {
    pub before: usize,
    pub after: usize,
}

impl SectionStat {
    pub fn saved(&self) -> usize {
        self.before.saturating_sub(self.after)
    }
    pub fn percent(&self) -> f64 {
        if self.before == 0 {
            0.0
        } else {
            self.saved() as f64 / self.before as f64 * 100.0
        }
    }
}

pub struct OptimizationReport {
    pub payload: String,
    pub initial_tokens: usize,
    pub optimized_tokens: usize,
    pub tokenizer_backend: String,
    pub exact: bool,
    pub guardrail_tokens: usize,
    pub sections: BTreeMap<String, SectionStat>,
}

impl OptimizationReport {
    pub fn tokens_saved(&self) -> i64 {
        self.initial_tokens as i64 - self.optimized_tokens as i64
    }

    pub fn percent_saved(&self) -> f64 {
        if self.initial_tokens == 0 {
            0.0
        } else {
            self.tokens_saved() as f64 / self.initial_tokens as f64 * 100.0
        }
    }

    pub fn compression_ratio(&self) -> f64 {
        if self.optimized_tokens == 0 {
            0.0
        } else {
            self.initial_tokens as f64 / self.optimized_tokens as f64
        }
    }
}

pub struct TokenDefenseProxy {
    model: String,
    guardrail: String,
    counter: TokenCounter,
}

impl TokenDefenseProxy {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            guardrail: CAVEMAN_GUARDRAIL.to_string(),
            counter: TokenCounter::new(model),
        }
    }

    fn stat(&self, before: &str, after: &str) -> SectionStat {
        SectionStat { before: self.counter.count(before), after: self.counter.count(after) }
    }

    /// Compress every input, inject the guardrail, and measure the savings.
    pub fn optimize_payload(
        &self,
        raw_code: &str,
        raw_logs: &str,
        system_data: Option<&Value>,
        task: &str,
    ) -> OptimizationReport {
        let raw_json = system_data
            .map(|v| serde_json::to_string_pretty(v).unwrap_or_default())
            .unwrap_or_default();

        let baseline: Vec<&str> = [task, raw_code, raw_logs, raw_json.as_str()]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect();
        let initial_tokens = self.counter.count(&baseline.join("\n\n"));

        let clean_code = if raw_code.is_empty() {
            String::new()
        } else {
            lossless_code_compressor(raw_code, &CodeOptions::default())
        };
        let clean_logs = if raw_logs.is_empty() {
            String::new()
        } else {
            lossless_terminal_cleaner(raw_logs, true, true)
        };
        let clean_data = system_data.map(json_to_minimal_yaml).unwrap_or_default();

        let mut sections = BTreeMap::new();
        if !raw_code.is_empty() {
            sections.insert("code".to_string(), self.stat(raw_code, &clean_code));
        }
        if !raw_logs.is_empty() {
            sections.insert("logs".to_string(), self.stat(raw_logs, &clean_logs));
        }
        if !raw_json.is_empty() {
            sections.insert("system_data".to_string(), self.stat(&raw_json, &clean_data));
        }

        let mut blocks = vec![self.guardrail.clone()];
        if !task.is_empty() {
            blocks.push(format!("TASK\n{}", task.trim()));
        }
        if !clean_code.is_empty() {
            blocks.push(format!("CODE\n{}", clean_code));
        }
        if !clean_logs.is_empty() {
            blocks.push(format!("LOGS\n{}", clean_logs));
        }
        if !clean_data.is_empty() {
            blocks.push(format!("STATE\n{}", clean_data));
        }
        let payload = blocks.join("\n\n");

        let optimized_tokens = self.counter.count(&payload);

        OptimizationReport {
            payload,
            initial_tokens,
            optimized_tokens,
            tokenizer_backend: self.counter.backend().to_string(),
            exact: self.counter.is_exact(),
            guardrail_tokens: self.counter.count(&self.guardrail),
            sections,
        }
    }

    pub fn print_summary(&self, r: &OptimizationReport) {
        let bar = "=".repeat(58);
        let dash = "-".repeat(58);
        println!("\n{}", bar);
        println!(" LOSSLESS TOKEN DEFENSE PROXY - OPTIMIZATION REPORT");
        println!("{}", bar);
        println!(" Model      : {}", self.model);
        println!(" Tokenizer  : {}", r.tokenizer_backend);
        if !r.exact {
            println!("              (approximate - exact BPE table unavailable)");
        }
        println!("{}", dash);

        if !r.sections.is_empty() {
            println!(" {:<14}{:>10}{:>10}{:>10}{:>8}", "SECTION", "BEFORE", "AFTER", "SAVED", "CUT");
            for (name, s) in &r.sections {
                println!(
                    " {:<14}{:>10}{:>10}{:>10}{:>7.1}%",
                    name, s.before, s.after, s.saved(), s.percent()
                );
            }
            println!("{}", dash);
        }

        println!(" Initial Tokens   : {:>10}", r.initial_tokens);
        println!(" Optimized Tokens : {:>10}", r.optimized_tokens);
        println!("   (of which guardrail overhead: {})", r.guardrail_tokens);
        println!(" Tokens Saved     : {:>10}  ({:.2}%)", r.tokens_saved(), r.percent_saved());
        println!(" Compression      : {:>10.2}x", r.compression_ratio());
        println!("{}", dash);

        let pct = r.percent_saved().clamp(0.0, 100.0);
        let filled = (pct / 100.0 * 40.0) as usize;
        println!(" [{}{}] {:.1}% cut", "#".repeat(filled), ".".repeat(40 - filled), pct);
        println!("{}\n", bar);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn pipeline_saves_tokens() {
        let proxy = TokenDefenseProxy::new("gpt-4o");
        let code = "# a comment\nx = 1\n\n\n\n\ny = 2  # trailing\n";
        let logs = "[1/250] Compiling a.c\n 45% completed\nerror: boom\n";
        let data = json!({"a": {"b": [1, 2, 3]}});
        let r = proxy.optimize_payload(code, logs, Some(&data), "fix it");
        assert!(r.initial_tokens > 0);
        for (name, s) in &r.sections {
            assert!(s.saved() > 0, "{} did not shrink", name);
        }
        assert!(r.payload.contains("OUTPUT PROTOCOL"));
        assert!(r.payload.contains("error: boom"));
    }

    #[test]
    fn empty_input_is_safe() {
        let proxy = TokenDefenseProxy::new("gpt-4o");
        let r = proxy.optimize_payload("", "", None, "");
        assert_eq!(r.initial_tokens, 0);
        assert_eq!(r.percent_saved(), 0.0);
    }
}
