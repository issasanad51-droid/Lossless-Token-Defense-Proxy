//! Pipeline-Filter execution fabric.
//!
//! Every lossless transform in the engine implements [`TokenFilter`]. The
//! [`Pipeline`] owns an ordered chain of those filters and records per-stage
//! telemetry so the orchestrator can attribute every saved token.

use std::fmt;
use std::time::Instant;

/// Lossless, pure transform over a UTF-8 payload.
///
/// Implementations MUST be semantically lossless: stripping comments,
/// collapsing progress tickers, or rewriting JSON as quote-free YAML is
/// permitted; mutating identifiers, literals, or control flow is not.
pub trait TokenFilter: Send + Sync {
    /// Apply the filter to `input` and return the rewritten payload.
    fn filter(&self, input: &str) -> String;

    /// Stable, human-readable stage name used in telemetry.
    fn name(&self) -> &'static str;
}

/// Per-stage measurement captured while a pipeline executes.
#[derive(Debug, Clone)]
pub struct StageReport {
    pub name: &'static str,
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub elapsed_us: u128,
}

impl StageReport {
    pub fn bytes_saved(&self) -> isize {
        self.input_bytes as isize - self.output_bytes as isize
    }
}

/// Full trace of one pipeline execution.
#[derive(Debug, Clone)]
pub struct PipelineTrace {
    pub input: String,
    pub output: String,
    pub stages: Vec<StageReport>,
}

impl PipelineTrace {
    pub fn input_bytes(&self) -> usize {
        self.input.len()
    }

    pub fn output_bytes(&self) -> usize {
        self.output.len()
    }
}

/// Ordered chain of [`TokenFilter`] stages.
pub struct Pipeline {
    filters: Vec<Box<dyn TokenFilter>>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    /// Append a filter to the tail of the chain. Returns `self` for fluency.
    pub fn register<F>(&mut self, filter: F) -> &mut Self
    where
        F: TokenFilter + 'static,
    {
        self.filters.push(Box::new(filter));
        self
    }

    /// Number of registered stages.
    pub fn len(&self) -> usize {
        self.filters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    /// Names of the registered stages, in execution order.
    pub fn stage_names(&self) -> Vec<&'static str> {
        self.filters.iter().map(|f| f.name()).collect()
    }

    /// Run every stage and return only the final payload.
    pub fn execute(&self, input: &str) -> String {
        self.execute_traced(input).output
    }

    /// Run every stage and keep the full telemetry trace.
    pub fn execute_traced(&self, input: &str) -> PipelineTrace {
        let mut current = input.to_string();
        let mut stages = Vec::with_capacity(self.filters.len());

        for filter in &self.filters {
            let before = current.len();
            let started = Instant::now();
            current = filter.filter(&current);
            stages.push(StageReport {
                name: filter.name(),
                input_bytes: before,
                output_bytes: current.len(),
                elapsed_us: started.elapsed().as_micros(),
            });
        }

        PipelineTrace {
            input: input.to_string(),
            output: current,
            stages,
        }
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pipeline")
            .field("stages", &self.stage_names())
            .finish()
    }
}

/// Identity filter used by the self-test harness to prove pipeline wiring.
pub struct IdentityFilter;

impl TokenFilter for IdentityFilter {
    fn filter(&self, input: &str) -> String {
        input.to_string()
    }

    fn name(&self) -> &'static str {
        "identity"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AppendFilter(&'static str, &'static str);

    impl TokenFilter for AppendFilter {
        fn filter(&self, input: &str) -> String {
            let mut out = input.to_string();
            out.push_str(self.1);
            out
        }
        fn name(&self) -> &'static str {
            self.0
        }
    }

    #[test]
    fn pipeline_applies_filters_in_order() {
        let mut p = Pipeline::new();
        p.register(AppendFilter("a", "A"))
            .register(AppendFilter("b", "B"));
        assert_eq!(p.execute("x"), "xAB");
        let trace = p.execute_traced("x");
        assert_eq!(trace.stages.len(), 2);
        assert_eq!(trace.stages[0].name, "a");
        assert_eq!(trace.stages[1].name, "b");
    }
}
