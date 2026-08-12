"""Tests for the hybrid structural retrieval layer.

The bar here is the same as the compression tests: assert real properties
(convergence, invariants, ordering guarantees), not just "it returned
something".
"""

from __future__ import annotations

import textwrap

import pytest

from retrieval.ast_graph import CodeGraph, CodeSpan, EdgeKind, build_graph
from retrieval.hybrid import HybridRetriever
from retrieval.ppr import personalized_pagerank, ppr_rank
from retrieval.rrf import CodeBlock, RankedList, fuse_ranks, reciprocal_rank_fusion
from retrieval.vector_store import VectorStore, code_tokenize

def succ(graph, node_id, kind=None):
    """Successor ids, optionally filtered by edge kind."""
    return [dst for dst, k, _w in graph.successors(node_id) if kind is None or k == kind]


def pred(graph, node_id, kind=None):
    return [src for src, k, _w in graph.predecessors(node_id) if kind is None or k == kind]


SAMPLE = {
    "auth.py": textwrap.dedent(
        '''
        """Auth helpers."""
        import hashlib


        class TokenValidator:
            """Validates bearer tokens."""

            def __init__(self, secret):
                self.secret = secret

            def validate(self, token):
                digest = self._hash(token)
                return digest == self.secret

            def _hash(self, token):
                return hashlib.sha256(token.encode()).hexdigest()


        class AdminValidator(TokenValidator):
            def validate(self, token):
                return super().validate(token) and self.is_admin(token)

            def is_admin(self, token):
                return token.startswith("admin-")
        '''
    ).strip(),
    "server.py": textwrap.dedent(
        '''
        from auth import TokenValidator


        def handle_request(request):
            validator = TokenValidator("secret")
            if not validator.validate(request.token):
                return reject(request)
            return accept(request)


        def reject(request):
            return {"status": 401}


        def accept(request):
            return {"status": 200}
        '''
    ).strip(),
}


# --------------------------------------------------------------------------
# AST graph
# --------------------------------------------------------------------------


class TestCodeGraph:
    def test_indexes_classes_and_methods(self):
        g = build_graph(SAMPLE)
        assert "auth.py::TokenValidator" in g.spans
        assert "auth.py::TokenValidator.validate" in g.spans
        assert "server.py::handle_request" in g.spans

    def test_method_kind_differs_from_function(self):
        g = build_graph(SAMPLE)
        assert g.spans["auth.py::TokenValidator.validate"].kind == "method"
        assert g.spans["server.py::handle_request"].kind == "function"

    def test_containment_edges(self):
        g = build_graph(SAMPLE)
        kids = succ(g, "auth.py::TokenValidator", EdgeKind.CONTAINS)
        assert "auth.py::TokenValidator.validate" in kids

    def test_inheritance_edge(self):
        g = build_graph(SAMPLE)
        parents = succ(g, "auth.py::AdminValidator", EdgeKind.INHERITS)
        assert "auth.py::TokenValidator" in parents

    def test_call_edge_resolves_unique_name(self):
        g = build_graph(SAMPLE)
        callees = succ(g, "server.py::handle_request", EdgeKind.CALLS)
        assert "server.py::reject" in callees
        assert "server.py::accept" in callees

    def test_ambiguous_call_is_not_guessed(self):
        """`validate` exists on two classes, so no call edge may be invented."""
        g = build_graph(SAMPLE)
        callees = succ(g, "server.py::handle_request", EdgeKind.CALLS)
        assert not any(c.endswith("::TokenValidator.validate") for c in callees)

    def test_predecessors_are_inverse_of_successors(self):
        g = build_graph(SAMPLE)
        for (src, dst, _kind) in g.edges:
            assert dst in succ(g, src)
            assert src in pred(g, dst)

    def test_line_ranges_are_within_file(self):
        g = build_graph(SAMPLE)
        for node_id, span in g.spans.items():
            total = len(SAMPLE[span.path].splitlines())
            assert 1 <= span.start_line <= span.end_line <= total, node_id

    def test_child_span_nested_in_parent_span(self):
        g = build_graph(SAMPLE)
        parent = g.spans["auth.py::TokenValidator"]
        child = g.spans["auth.py::TokenValidator.validate"]
        assert parent.start_line <= child.start_line
        assert child.end_line <= parent.end_line

    def test_syntax_error_is_skipped_not_fatal(self):
        g = build_graph({"good.py": "def f():\n    pass\n", "bad.py": "def ("})
        assert "good.py::f" in g.spans
        assert not any(s.path == "bad.py" and s.kind != "module" for s in g.spans.values())

    def test_decorators_included_in_span(self):
        src = "@cache\n@retry(3)\ndef fetch():\n    return 1\n"
        g = build_graph({"m.py": src})
        assert g.spans["m.py::fetch"].start_line == 1

    def test_overlaps(self):
        a = CodeSpan("a", "function", "a", "a", "a.py", 1, 10)
        b = CodeSpan("b", "function", "b", "b", "a.py", 5, 15)
        c = CodeSpan("c", "function", "c", "c", "a.py", 11, 20)
        d = CodeSpan("d", "function", "d", "d", "b.py", 1, 10)
        assert a.overlaps(b) and b.overlaps(a)
        assert not a.overlaps(c)
        assert not a.overlaps(d)  # different files never overlap


# --------------------------------------------------------------------------
# Tokenizer + vector store
# --------------------------------------------------------------------------


class TestCodeTokenize:
    def test_splits_snake_case(self):
        assert "token" in code_tokenize("validate_token")
        assert "validate" in code_tokenize("validate_token")

    def test_splits_camel_case(self):
        toks = code_tokenize("TokenValidator")
        assert "token" in toks and "validator" in toks

    def test_keeps_compound_form(self):
        assert "validate_token" in code_tokenize("validate_token")

    def test_drops_single_chars_and_stopwords(self):
        toks = code_tokenize("def x(self): return self")
        assert "x" not in toks
        assert "self" not in toks


class TestVectorStore:
    def test_exact_term_ranks_first(self):
        vs = VectorStore()
        vs.add("a", "compute sha256 digest of the bearer token")
        vs.add("b", "render html templates for the dashboard")
        assert vs.search("sha256 digest")[0][0] == "a"

    def test_scores_are_bounded_cosine(self):
        vs = VectorStore()
        vs.add_many({"a": "alpha beta gamma", "b": "beta gamma delta"})
        for _, score in vs.search("beta"):
            assert -1e-9 <= score <= 1.0 + 1e-9

    def test_unknown_term_returns_nothing(self):
        vs = VectorStore()
        vs.add("a", "alpha beta")
        assert vs.search("zzzznotpresent") == []

    def test_deterministic_tie_break(self):
        vs = VectorStore()
        vs.add_many({"b": "same words here", "a": "same words here"})
        assert [d for d, _ in vs.search("same words here")] == ["a", "b"]

    def test_idf_downweights_ubiquitous_terms(self):
        """A term in every document carries no discriminative signal."""
        vs = VectorStore()
        for i in range(5):
            vs.add(f"d{i}", "common token here")
        vs.add("rare", "common token unicorn")
        assert vs.search("unicorn")[0][0] == "rare"

    def test_custom_embedder_is_used(self):
        calls = []

        def embedder(text):
            calls.append(text)
            return [1.0, 0.0] if "yes" in text else [0.0, 1.0]

        vs = VectorStore(embedder=embedder)
        vs.add_many({"y": "yes doc", "n": "no doc"})
        assert vs.search("yes")[0][0] == "y"
        assert calls, "embedder was never invoked"


# --------------------------------------------------------------------------
# Personalized PageRank
# --------------------------------------------------------------------------


class TestPPR:
    def test_scores_form_probability_distribution(self):
        g = build_graph(SAMPLE)
        scores = personalized_pagerank(g, {"server.py::handle_request": 1.0})
        assert scores
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(v >= 0 for v in scores.values())

    def test_seed_dominates(self):
        g = build_graph(SAMPLE)
        seed = "auth.py::TokenValidator._hash"
        scores = personalized_pagerank(g, {seed: 1.0})
        top = max(scores.items(), key=lambda kv: kv[1])[0]
        assert top == seed

    def test_different_seeds_give_different_rankings(self):
        """If personalization did nothing this would be global PageRank."""
        g = build_graph(SAMPLE)
        a = personalized_pagerank(g, {"server.py::reject": 1.0})
        b = personalized_pagerank(g, {"auth.py::AdminValidator.is_admin": 1.0})
        assert a != b

    def test_empty_seeds_yield_empty(self):
        g = build_graph(SAMPLE)
        assert personalized_pagerank(g, {}) == {}

    def test_unknown_seed_yields_empty(self):
        g = build_graph(SAMPLE)
        assert personalized_pagerank(g, {"nope.py::ghost": 1.0}) == {}

    def test_neighbours_outrank_strangers(self):
        g = build_graph(SAMPLE)
        scores = personalized_pagerank(g, {"server.py::handle_request": 1.0})
        assert scores.get("server.py::reject", 0) > scores.get("auth.py::AdminValidator.is_admin", 0)

    def test_converges_before_iteration_cap(self):
        g = build_graph(SAMPLE)
        loose = personalized_pagerank(g, {"server.py::handle_request": 1.0}, max_iterations=200)
        tight = personalized_pagerank(g, {"server.py::handle_request": 1.0}, max_iterations=1000)
        for node in loose:
            assert loose[node] == pytest.approx(tight[node], abs=1e-6)

    def test_damping_changes_spread(self):
        g = build_graph(SAMPLE)
        seed = {"server.py::handle_request": 1.0}
        low = personalized_pagerank(g, seed, damping=0.1)
        high = personalized_pagerank(g, seed, damping=0.95)
        # Low damping teleports home constantly -> more mass stays on the seed.
        assert low["server.py::handle_request"] > high["server.py::handle_request"]

    def test_ppr_rank_sorted_descending(self):
        g = build_graph(SAMPLE)
        ranked = ppr_rank(g, {"server.py::handle_request": 1.0}, top_k=5)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)
        assert len(ranked) <= 5

    def test_exclude_seeds(self):
        g = build_graph(SAMPLE)
        seed = "server.py::handle_request"
        ranked = ppr_rank(g, {seed: 1.0}, top_k=10, exclude_seeds=True)
        assert seed not in [n for n, _ in ranked]


# --------------------------------------------------------------------------
# RRF
# --------------------------------------------------------------------------


class TestRRF:
    def test_formula_matches_definition(self):
        lists = [RankedList("r1", [("a", 9.0), ("b", 1.0)])]
        fused = fuse_ranks(lists, k=60)
        assert fused["a"][0] == pytest.approx(1 / 61)
        assert fused["b"][0] == pytest.approx(1 / 62)

    def test_raw_scores_are_ignored(self):
        """RRF must depend only on order - that is the whole point."""
        a = fuse_ranks([RankedList("r", [("x", 1000.0), ("y", 999.0)])])
        b = fuse_ranks([RankedList("r", [("x", 0.02), ("y", 0.01)])])
        assert a["x"][0] == pytest.approx(b["x"][0])

    def test_agreement_beats_a_single_top_hit(self):
        lists = [
            RankedList("r1", [("solo", 1.0), ("agreed", 1.0)]),
            RankedList("r2", [("other", 1.0), ("agreed", 1.0)]),
        ]
        fused = fuse_ranks(lists)
        assert fused["agreed"][0] > fused["solo"][0]

    def test_provenance_recorded(self):
        lists = [RankedList("r1", [("a", 1.0)]), RankedList("r2", [("z", 1.0), ("a", 1.0)])]
        _, prov = fuse_ranks(lists)["a"]
        assert prov == {"r1": 1, "r2": 2}

    def test_zero_weight_list_is_skipped(self):
        lists = [RankedList("on", [("a", 1.0)]), RankedList("off", [("b", 1.0)], weight=0.0)]
        assert "b" not in fuse_ranks(lists)

    # -- grouping ----------------------------------------------------------

    def _graph_with_spans(self, spans):
        g = CodeGraph()
        for s in spans:
            g.add_span(s)
        return g

    def test_overlapping_spans_are_merged(self):
        g = self._graph_with_spans(
            [
                CodeSpan("a", "function", "a", "a", "f.py", 10, 20),
                CodeSpan("b", "function", "b", "b", "f.py", 15, 25),
            ]
        )
        blocks = reciprocal_rank_fusion(
            [RankedList("r", [("a", 1.0), ("b", 1.0)])], g, top_n=3, drop_enclosing=False
        )
        assert len(blocks) == 1
        assert (blocks[0].start_line, blocks[0].end_line) == (10, 25)

    def test_distant_spans_stay_separate(self):
        g = self._graph_with_spans(
            [
                CodeSpan("a", "function", "a", "a", "f.py", 1, 5),
                CodeSpan("b", "function", "b", "b", "f.py", 400, 410),
            ]
        )
        blocks = reciprocal_rank_fusion([RankedList("r", [("a", 1.0), ("b", 1.0)])], g, top_n=3)
        assert len(blocks) == 2

    def test_spans_in_different_files_never_merge(self):
        g = self._graph_with_spans(
            [
                CodeSpan("a", "function", "a", "a", "one.py", 10, 20),
                CodeSpan("b", "function", "b", "b", "two.py", 10, 20),
            ]
        )
        blocks = reciprocal_rank_fusion([RankedList("r", [("a", 1.0), ("b", 1.0)])], g, top_n=3)
        assert len(blocks) == 2

    def test_merge_gap_bridges_adjacent_spans(self):
        g = self._graph_with_spans(
            [
                CodeSpan("a", "function", "a", "a", "f.py", 1, 10),
                CodeSpan("b", "function", "b", "b", "f.py", 12, 20),
            ]
        )
        joined = reciprocal_rank_fusion([RankedList("r", [("a", 1.0), ("b", 1.0)])], g, merge_gap=2)
        split = reciprocal_rank_fusion([RankedList("r", [("a", 1.0), ("b", 1.0)])], g, merge_gap=0)
        assert len(joined) == 1
        assert len(split) == 2

    def test_enclosing_span_is_dropped(self):
        """A module span must not drag the whole file into the result."""
        g = self._graph_with_spans(
            [
                CodeSpan("mod", "module", "f", "f", "f.py", 1, 500),
                CodeSpan("fn", "function", "fn", "fn", "f.py", 10, 20),
            ]
        )
        blocks = reciprocal_rank_fusion([RankedList("r", [("mod", 1.0), ("fn", 1.0)])], g, top_n=3)
        assert len(blocks) == 1
        assert blocks[0].line_count == 11

    def test_enclosing_score_is_inherited_not_lost(self):
        g = self._graph_with_spans(
            [
                CodeSpan("mod", "module", "f", "f", "f.py", 1, 500),
                CodeSpan("fn", "function", "fn", "fn", "f.py", 10, 20),
            ]
        )
        with_parent = reciprocal_rank_fusion(
            [RankedList("r", [("fn", 1.0), ("mod", 1.0)])], g, top_n=1
        )[0]
        alone = reciprocal_rank_fusion([RankedList("r", [("fn", 1.0)])], g, top_n=1)[0]
        assert with_parent.fused_score > alone.fused_score

    def test_returns_at_most_top_n(self):
        spans = [CodeSpan(f"n{i}", "function", f"n{i}", f"n{i}", "f.py", i * 50, i * 50 + 5)
                 for i in range(1, 11)]
        g = self._graph_with_spans(spans)
        ranking = [(s.node_id, 1.0) for s in spans]
        assert len(reciprocal_rank_fusion([RankedList("r", ranking)], g, top_n=3)) == 3

    def test_density_ordering_is_monotonic(self):
        spans = [CodeSpan(f"n{i}", "function", f"n{i}", f"n{i}", "f.py", i * 50, i * 50 + i * 3)
                 for i in range(1, 8)]
        g = self._graph_with_spans(spans)
        ranking = [(s.node_id, 1.0) for s in spans]
        blocks = reciprocal_rank_fusion([RankedList("r", ranking)], g, top_n=7)
        densities = [b.density for b in blocks]
        assert densities == sorted(densities, reverse=True)

    def test_rank_by_score_prefers_larger_regions(self):
        g = self._graph_with_spans(
            [
                CodeSpan("small", "function", "s", "s", "f.py", 1, 3),
                CodeSpan("big", "function", "b", "b", "g.py", 1, 200),
            ]
        )
        lists = [RankedList("r", [("big", 1.0), ("small", 1.0)])]
        by_score = reciprocal_rank_fusion(lists, g, top_n=1, rank_by="score")[0]
        by_density = reciprocal_rank_fusion(lists, g, top_n=1, rank_by="density")[0]
        assert by_score.node_ids == ["big"]
        assert by_density.node_ids == ["small"]

    def test_max_block_lines_filters(self):
        g = self._graph_with_spans(
            [
                CodeSpan("small", "function", "s", "s", "f.py", 1, 3),
                CodeSpan("big", "function", "b", "b", "g.py", 1, 900),
            ]
        )
        blocks = reciprocal_rank_fusion(
            [RankedList("r", [("big", 1.0), ("small", 1.0)])], g, top_n=3, max_block_lines=100
        )
        assert [b.node_ids for b in blocks] == [["small"]]

    def test_unknown_nodes_are_ignored(self):
        g = self._graph_with_spans([CodeSpan("a", "function", "a", "a", "f.py", 1, 5)])
        blocks = reciprocal_rank_fusion([RankedList("r", [("ghost", 1.0), ("a", 1.0)])], g, top_n=3)
        assert [b.node_ids for b in blocks] == [["a"]]

    def test_empty_input_returns_empty(self):
        assert reciprocal_rank_fusion([], CodeGraph(), top_n=3) == []

    def test_bad_rank_by_raises(self):
        g = self._graph_with_spans([CodeSpan("a", "function", "a", "a", "f.py", 1, 5)])
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([RankedList("r", [("a", 1.0)])], g, rank_by="nope")

    def test_primary_label_matches_first_symbol_in_block(self):
        """The label must name the symbol the rendered text actually starts at."""
        g = self._graph_with_spans(
            [
                CodeSpan("second", "function", "second", "second", "f.py", 20, 30),
                CodeSpan("first", "function", "first", "first", "f.py", 10, 18),
            ]
        )
        block = reciprocal_rank_fusion(
            [RankedList("r", [("second", 1.0), ("first", 1.0)])], g, top_n=1, merge_gap=5
        )[0]
        assert block.start_line == 10
        assert block.primary.startswith("first")
        assert block.symbols == ["first", "second"]

    def test_block_text_extracts_exact_lines(self):
        src = "\n".join(f"line{i}" for i in range(1, 11))
        block = CodeBlock(path="f.py", start_line=3, end_line=5)
        assert block.text({"f.py": src}) == "line3\nline4\nline5"


# --------------------------------------------------------------------------
# End-to-end hybrid
# --------------------------------------------------------------------------


class TestHybridRetriever:
    @staticmethod
    @pytest.fixture(scope="class")
    def retriever():
        return HybridRetriever(SAMPLE)

    def test_all_three_retrievers_report(self, retriever):
        result = retriever.retrieve("validate a bearer token")
        assert set(result.rankings) == {"vector", "ast", "ppr"}
        assert not result.errors

    def test_returns_top_three_blocks(self, retriever):
        assert len(retriever.retrieve("validate token").blocks) <= 3

    def test_finds_the_relevant_symbol(self, retriever):
        result = retriever.retrieve("hash a token with sha256")
        joined = " ".join(nid for b in result.blocks for nid in b.node_ids)
        assert "_hash" in joined

    def test_blocks_never_overlap_each_other(self, retriever):
        """The anti-duplication guarantee: no line is sent twice."""
        blocks = retriever.retrieve("validate admin token").blocks
        for i, a in enumerate(blocks):
            for b in blocks[i + 1:]:
                if a.path == b.path:
                    assert a.end_line < b.start_line or b.end_line < a.start_line

    def test_context_is_real_source(self, retriever):
        result = retriever.retrieve("reject unauthorized request")
        context = result.to_context(SAMPLE)
        assert context
        for line in context.splitlines():
            if line.startswith("# ") and ":" in line:
                continue
            if line.strip():
                assert line in SAMPLE["auth.py"] or line in SAMPLE["server.py"]

    def test_context_is_smaller_than_whole_corpus(self, retriever):
        result = retriever.retrieve("validate token")
        whole = sum(len(s) for s in SAMPLE.values())
        assert len(result.to_context(SAMPLE)) < whole

    def test_timings_recorded_for_each_stage(self, retriever):
        result = retriever.retrieve("token")
        for stage in ("vector", "ast", "ppr", "fusion", "seeding"):
            assert stage in result.timings_ms

    def test_deterministic_across_runs(self, retriever):
        """Thread scheduling must not leak into the output."""
        a = retriever.retrieve("validate the bearer token")
        b = retriever.retrieve("validate the bearer token")
        key = lambda r: [(x.path, x.start_line, x.end_line) for x in r.blocks]
        assert key(a) == key(b)

    def test_one_failing_retriever_does_not_sink_the_query(self, retriever):
        broken = HybridRetriever(SAMPLE)

        def boom(*_args, **_kwargs):
            raise RuntimeError("embedding service down")

        broken._vector_search = boom
        result = broken.retrieve("validate token", seed_nodes={"auth.py::TokenValidator": 1.0})
        assert "vector" in result.errors
        assert result.blocks, "AST + PPR should still produce results"

    def test_empty_query_is_safe(self, retriever):
        assert retriever.retrieve("").blocks == []

    def test_nonsense_query_does_not_crash(self, retriever):
        retriever.retrieve("zzzq xxqq vvbb")  # must not raise

    def test_explain_mentions_contributors(self, retriever):
        text = retriever.retrieve("validate token").explain()
        assert "via" in text and "validate token" in text

    def test_stats(self, retriever):
        stats = retriever.stats()
        assert stats["nodes"] > 0 and stats["indexed_docs"] > 0

    def test_empty_corpus(self):
        empty = HybridRetriever({})
        assert empty.retrieve("anything").blocks == []
