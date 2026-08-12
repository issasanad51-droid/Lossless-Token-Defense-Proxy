# Lossless Token Defense Proxy

Middleware that intercepts raw data on its way to an LLM, strips token waste
**losslessly**, and reports the exact savings with `tiktoken`.

Two implementations, kept in lockstep by a parity check in CI:

| | Path | Run |
|---|---|---|
| Python (reference) | repo root | `python orchestrator.py` |
| Rust (port) | `rust/` | `cargo run --release` |

---

## Quick start

```bash
pip install -r requirements.txt
python orchestrator.py                 # demo with dummy data
python orchestrator.py --show-payload  # also print the optimized payload
python orchestrator.py --json          # machine-readable report
pytest -q                              # 53 losslessness tests
```

Sample output:

```
==========================================================
 LOSSLESS TOKEN DEFENSE PROXY - OPTIMIZATION REPORT
==========================================================
 Model      : gpt-4o
 Tokenizer  : tiktoken:o200k_base
----------------------------------------------------------
 SECTION           BEFORE     AFTER     SAVED     CUT
 code                 458       262       196   42.8%
 logs                 449       147       302   67.3%
 system_data          385       287        98   25.5%
----------------------------------------------------------
 Initial Tokens   :      1,306
 Optimized Tokens :        932
   (of which guardrail overhead: 212)
 Tokens Saved     :        374  (28.64%)
 Compression      :       1.40x
----------------------------------------------------------
 [###########.............................] 28.6% cut
==========================================================
```

> The headline percentage is *after* paying ~212 tokens for the guardrail.
> Payload compression alone is ~45%; the guardrail buys that back on the
> **output** side, which is where the expensive tokens usually are.

---

## What "lossless" means here

Not "looks shorter" — the pipeline is designed so meaning is provably preserved,
and the test suite enforces it:

| Module | Guarantee | Enforced by |
|---|---|---|
| `compressors.lossless_code_compressor` | compressed source parses to an **identical AST** | `ast.dump()` equality on every sample **+ 400 real stdlib files** |
| `data_converter.json_to_minimal_yaml` | flattened text **round-trips** to the exact original object | `minimal_yaml_to_json()` inverse on 12 property cases |
| `compressors.lossless_terminal_cleaner` | every error / failure / completion line survives | explicit signal assertions |

Verified stress result: **400/400 Python stdlib files compressed to a
byte-identical AST, 22.4% smaller.**

---

## Modules

### `compressors.py`

`lossless_code_compressor(raw_code)` — line-by-line parse with string-state
tracking.

* removes whole-line and trailing `#` comments
* collapses runs of blank lines (configurable)
* **never** touches a `#` inside a string — `note = "refund # not a comment"` is safe
* **never** alters indentation or docstrings, including blank lines inside them
* **keeps** non-prose comments, because removing them would be lossy:
  shebangs, coding cookies, `# noqa`, `# type:`, `# fmt: off`, `# pylint`, ...

`lossless_terminal_cleaner(raw_logs)` — regex noise filter where **signal always
wins**.

* drops `[1/250] Compiling ...`, `45% completed`, `4.2MB/10MB 3.4MB/s ETA 00:02`,
  spinners, progress bars, apt/npm/git chatter
* keeps errors, fatals, panics, tracebacks, undefined references, exit codes,
  completion markers
* a line matching both (`[42/250] error: ...`) is **kept** — signal beats noise
* collapses `\r` redraw frames, strips ANSI escapes, folds duplicates into `(xN)`
* annotates removals as `... 47 progress lines omitted ...` so nothing vanishes silently

### `data_converter.py`

`json_to_minimal_yaml(data_obj)` flattens nested JSON to bracket-free,
quote-free dotted paths:

```
service.name payment-api
service.replicas 6
service.healthy true
runtime.feature_flags.async_refunds true
dependencies.0.version "15.4"
open_ports 8080, 8443, 9090
```

Quoting is **minimal but not absent**: `"15.4"` keeps its quotes because
dropping them would turn a string into a float on the way back. That single
detail is the difference between compression and corruption.
`minimal_yaml_to_json()` is the inverse and proves it.

### `orchestrator.py`

`TokenDefenseProxy(model="gpt-4o").optimize_payload(raw_code, raw_logs, system_data)`

1. counts baseline tokens on what you *would* have pasted (raw code + raw logs + pretty JSON)
2. routes each input through its compressor
3. injects the **Ponytail/Caveman guardrail**
4. counts final tokens and prints the per-section summary

```python
from orchestrator import TokenDefenseProxy

report = TokenDefenseProxy(model="gpt-4o").optimize_payload(
    raw_code=my_code, raw_logs=my_logs, system_data=my_dict,
    task="Fix the link error.",
)
print(report.tokens_saved, report.percent_saved, report.payload)
```

### The guardrail

Compressing the prompt is half the job; a chatty reply costs more than the patch.
`CAVEMAN_GUARDRAIL` is a 10-rule output contract: unified diffs only, banned
openers/closers, no re-printing unchanged code, `ASK:` instead of guessing,
one optional `WHY:` line capped at 15 words.

---

## Tokenizer resilience

`tiktoken` downloads BPE tables on first use, which fails on air-gapped or
CDN-blocked machines. `tokenizer.py` degrades instead of crashing:

1. exact encoding for the model (`gpt-4o` → `o200k_base`)
2. any locally cached encoding, clearly labelled as a proxy
3. deterministic heuristic estimator, flagged as approximate

Savings percentages stay valid at every tier, because baseline and optimized
text are always measured with the same counter.

---

## Hybrid structural retrieval (`retrieval/`)

Compression makes each token cheaper. Retrieval decides which tokens get sent
at all — the larger win on a real codebase. Three retrievers run concurrently
and are blended with Reciprocal Rank Fusion:

```bash
python -m retrieval                          # demo over this repo
python -m retrieval "how are tokens counted" # your own query
```

```
vector lookup  ─┐
AST structure  ─┼─> RRF ─> overlap grouping ─> top 3 densest blocks
PPR            ─┘
```

| Stage | File | What it contributes |
|---|---|---|
| AST graph | `ast_graph.py` | Functions, classes, methods and the `CALLS`/`CONTAINS`/`INHERITS`/`IMPORTS` edges between them |
| Vector lookup | `vector_store.py` | TF-IDF cosine over identifier-aware tokens (recall) |
| Structural match | `hybrid.py` | Exact hits on names, signatures, decorators (precision) |
| PPR | `ppr.py` | Personalized PageRank seeded on the query's own hits, scoring structural importance of dependencies |
| Fusion | `rrf.py` | `1/(k+rank)` blend, overlap grouping, density ranking |

Design points worth knowing:

* **RRF needs no score calibration.** Cosine similarity, PageRank mass and
  lexical overlap live on incompatible scales; fusing *ranks* sidesteps that
  entirely, and cross-retriever agreement earns an explicit bonus.
* **Overlapping spans are merged, not stacked.** A class, its method and the
  enclosing module are three nodes covering the same lines. Enclosing spans are
  dropped and their score folded into the specific children, so a module hit
  can't drag a whole file into context. Returned blocks are guaranteed
  non-overlapping — no line is ever sent twice.
* **Density, not raw score.** Blocks rank by score per unit length with a
  sublinear penalty and a minimum-length floor, so a tight function beats both
  a sprawling module and a 2-line getter.
* **PPR is seeded from the query's own hits**, which is what makes it
  *personalized* rather than static global importance. Empty seeds return an
  empty result rather than silently falling back to global PageRank.
* **Failure isolation.** Each retriever runs in the thread pool; if one raises,
  the error is recorded in `result.errors` and the other two still answer.

On this repository a query returns ~3 blocks totalling 18–35 lines out of
~3,400 — **99%+ of the codebase skipped** — with `result.explain()` showing
which retrievers voted for each block and at what rank.

> **Two honest caveats.** (1) The vector stage is a *lexical* TF-IDF vector
> space, not a semantic embedding: no model could be downloaded in this
> sandbox. It is a real vector-space retriever with real cosine scoring, and
> `VectorStore(embedder=...)` swaps in any embedding function without touching
> the rest of the pipeline — but it matches words, not meaning. (2) The three
> retrievers run in threads. Here the work is CPU-bound Python, so threading
> overlaps rather than parallelizes it; the structure pays off once the vector
> stage becomes an I/O-bound call to an embedding API or vector DB.
>
> The AST layer is Python-only (it uses the `ast` module). Other languages need
> a per-language parser behind the same `CodeSpan`/`CodeGraph` interface.

---

## Rust port

Same algorithms and guarantees, in `rust/`:

```bash
cd rust
cargo test            # unit tests, incl. round-trip property tests
cargo run --release   # same demo, same report
```

`tiktoken-rs` embeds its BPE tables in the binary, so the Rust build needs no
network at runtime. CI runs `scripts/check_parity.py`, which diffs Rust output
against Python output on shared `fixtures/` — the two ports cannot silently drift.

> **Status: the Rust port has not been compiled yet.** The sandbox this was
> built in has no Rust toolchain and blocks crates.io, so unlike the Python side
> (run and tested here) the Rust code is review-quality, not build-verified.
> Expect to fix minor compile errors on first `cargo build`.

### Native retrieval ecosystem (`rust/src/indexer/`)

A dependency-free retrieval layer, built as three composable filters behind a
shared `Filter` / `PipelineContext` abstraction, chained by `RetrievalEngine`:

| Subsystem | File | Role |
| --- | --- | --- |
| A | `ast_chunker.rs` | Hand-written scope parsers for Rust, Python, JS/TS, Go |
| B | `ppr_graph.rs` | Personalized PageRank over the call/containment graph |
| C | `rrf_blender.rs` | Weighted reciprocal rank fusion + proximity merging |

No tree-sitter and no new crates — Phase 3 adds zero dependencies.

The chunker runs `mask_source` before any brace or indent counting, blanking
string and comment interiors while preserving length and line structure, so a
`}` inside a string can never corrupt a scope boundary and coordinates stay
exact. Chunks carry a 256-dim hashed bag-of-features embedding. **This is
lexical, not semantic** — there is no model behind it, and it will not match
synonyms.

Two tuning choices are deliberately non-textbook and should not be "corrected":

* **PPR damping is 0.6, not 0.85.** On a small, bidirectionally-traversed code
  graph, 0.85 lets the most *central* node outrank the seed, which silently
  turns personalized PageRank into global PageRank. `max_iterations` is 200
  because reaching `tol=1e-9` needs ~132 iterations, and the old cap of 100
  meant `converged` was never true.
* **Merged blocks take the max contributing score, never the sum**, so a block
  cannot win by absorbing neighbours.

`RetrievalEngine::query` drops PPR results scoring 0.0. PPR assigns a score to
every node, and nodes the walk never reached come back as exactly zero; passing
those to the blender still earns them a rank from array position, which drags
unrelated files into the answer.

### Enabling CI

The workflow lives at **`ci/github-actions-ci.yml`** instead of
`.github/workflows/` because the GitHub App used to push lacks the `workflows`
permission. To turn it on:

```bash
mkdir -p .github/workflows
git mv ci/github-actions-ci.yml .github/workflows/ci.yml
git commit -m "Enable CI" && git push
```

If you do this through the GitHub web UI instead, note that the filename box
resolves **relative to the file's current directory**. Renaming from inside
`ci/` to `.github/workflows/ci.yml` produces `ci/.github/workflows/ci.yml`,
which Actions ignores. Prefix with `../` to escape, or just run the commands
above locally.

It builds and tests Python on 3.9/3.11/3.12, builds and tests the Rust crate,
and runs the parity check — which is what will compile the Rust port for you.

**Which should you use?** Python, for now. The hot path is tokenization, and
`tiktoken`'s core is already compiled Rust; the pure-Python glue is microseconds
on realistic payloads. Rust earns its keep when this becomes a high-throughput
network proxy with tail-latency SLAs, not a preprocessing library.

---

## Layout

```
ci/                   GitHub Actions workflow (move to .github/workflows to enable)
compressors.py        code + terminal compressors
data_converter.py     JSON -> minimal YAML (+ inverse)
tokenizer.py          tiktoken wrapper with offline fallback
orchestrator.py       TokenDefenseProxy + runnable demo
test_lossless.py      53 tests proving losslessness
retrieval/            hybrid structural retrieval (AST graph + vectors + PPR + RRF)
test_retrieval.py     66 tests for the retrieval layer
fixtures/             shared parity inputs
scripts/check_parity.py
rust/                 Rust port (lib + demo binary + tests)
rust/src/indexer/     native retrieval: AST chunker + PPR graph + RRF blender
```

## Known limits

* The code compressor targets `#`-comment languages (Python, Ruby, shell, YAML).
  C-style `//` and `/* */` are not handled yet.
* A single-element scalar array round-trips to a bare scalar — the one
  documented ambiguity in the flattening format.
* Log filtering is heuristic. Signal patterns are deliberately broad (it keeps
  more than it drops when uncertain), so a line containing "error" survives even
  if it is noise.
* The retrieval AST layer in Python is Python-only; the Rust chunker covers
  Rust, Python, JS/TS and Go. Both are line-and-brace scanners, not full
  parsers — they handle the shapes real code takes, not every legal one.
* Rust chunk embeddings are hashed lexical features, not a learned model.
  Vector search there matches shared identifiers, not meaning.
