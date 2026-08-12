# Lossless Token Defense Proxy Engine

Ultra-low-latency **context retrieval optimizer**. Every core stage is a
custom `TokenFilter` or a first-party indexer — no generic RAG frameworks.

```
src/
├── main.rs                 # orchestration, telemetry, in-binary test runner
├── pipeline.rs             # TokenFilter trait + execution pipeline
├── filters/
│   ├── mod.rs              # Layer 1 & 2 registration + payload classifier
│   ├── code_cleaner.rs     # lossless comment / whitespace stripper
│   ├── log_cleaner.rs      # pre-compiled regex build-log compressor
│   └── data_converter.rs   # JSON → quote-free, brace-free YAML
└── indexer/
    ├── mod.rs              # RetrievalEngine façade
    ├── ast_parser.rs       # Tree-Sitter Point/Range scope mapping
    ├── ppr_graph.rs        # semantic-weighted Personalized PageRank
    └── rrf_blender.rs      # overlap-reconciling, confidence-calibrated RRF
```

## Pipeline

1. **Layer 1 filters** strip comments, tickers, and ANSI without touching
   program meaning. Indentation that carries semantics is preserved.
2. **Layer 2** rewrites nested JSON as quote-free block YAML. Key order
   and numeric lexemes survive; a built-in inverse proves losslessness.
3. **AST mapper** isolates each `struct` / `enum` / `class` / function
   body as **one** immutable chunk. Coordinates are `tree_sitter::Range`
   values. No sliding windows.
4. **PPR graph** wires `CALLS_FUNCTION` (3.0), `IMPORTS_MODULE` (2.0),
   and `HAS_PROPERTY` (1.0) edges, then walks from a personalization
   vector taken from vector-similarity hits.
5. **RRF blender** merges snippets that sit inside a 15-line window and
   floats any hit with confidence `> 0.95` above positional damping:
   `Score = Weight × (1 / (K + Rank))`.

## Run

```bash
cargo run --release
```

The binary prints the self-test harness, per-payload filter savings, the
retrieval ranking, and a `tiktoken-rs` (`cl100k_base`) telemetry block:

* Initial Token Size
* Final Optimized Token Size
* Total Lossless Tokens Saved & Saving Percentage

CI (`.github/workflows/rust.yml`) runs the same command on every push.
