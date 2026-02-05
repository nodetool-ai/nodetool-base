"""
Collection-pipeline workflow tests.

Exercises list manipulation, dictionary transforms, and numeric
aggregation nodes — all through run_graph_async with real graphs.
"""

import pytest

from nodetool.dsl.graph import create_graph, run_graph_async
from nodetool.dsl.nodetool.constant import (
    List as ConstList,
    Dict as ConstDict,
)
from nodetool.dsl.nodetool.list import (
    Average,
    Chunk,
    Dedupe,
    Difference,
    Extend,
    Flatten,
    Intersection,
    Length as ListLength,
    ListRange,
    Maximum,
    Minimum,
    Product,
    Reverse,
    SelectElements,
    Slice as ListSlice,
    Sort,
    Sum as ListSum,
    Union,
)
from nodetool.dsl.nodetool.dictionary import (
    ArgMax,
    Combine,
    Filter as DictFilter,
    GetValue,
    ParseJSON,
    Remove,
    ToJSON,
    ToYAML,
    Update,
    Zip,
)
from nodetool.dsl.nodetool.output import Output


# ======================================================================
# LIST PIPELINES
# ======================================================================


class TestInventoryPipeline:
    """Deduplicate and sort product-count data."""

    @pytest.mark.asyncio
    async def test_dedupe_and_sort(self):
        raw = ConstList(value=[4, 8, 2, 8, 4, 15, 2])
        unique = Dedupe(values=raw.output)
        ordered = Sort(values=unique.output)
        sink = Output(name="r", value=ordered.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == [2, 4, 8, 15]

    @pytest.mark.asyncio
    async def test_restock_stats(self):
        stock = ConstList(value=[0.0, 12.0, 3.0, 0.0, 7.0])
        total = ListSum(values=stock.output)
        low = Minimum(values=stock.output)
        o1 = Output(name="total", value=total.output)
        o2 = Output(name="low", value=low.output)
        bag = await run_graph_async(create_graph(o1, o2))
        assert bag["total"] == 22.0
        assert bag["low"] == 0.0


class TestGradePipeline:
    """Statistics for a classroom of grades."""

    @pytest.mark.asyncio
    async def test_grade_stats(self):
        grades = ConstList(value=[78.0, 92.0, 65.0, 88.0, 95.0, 71.0])
        avg = Average(values=grades.output)
        top = Maximum(values=grades.output)
        bottom = Minimum(values=grades.output)
        o1 = Output(name="avg", value=avg.output)
        o2 = Output(name="top", value=top.output)
        o3 = Output(name="bottom", value=bottom.output)
        bag = await run_graph_async(create_graph(o1, o2, o3))
        assert 80.0 < bag["avg"] < 83.0
        assert bag["top"] == 95.0
        assert bag["bottom"] == 65.0

    @pytest.mark.asyncio
    async def test_top_three_grades(self):
        grades = ConstList(value=[78.0, 92.0, 65.0, 88.0, 95.0, 71.0])
        ordered = Sort(values=grades.output)
        rev = Reverse(values=ordered.output)
        top3 = ListSlice(values=rev.output, start=0, stop=3)
        sink = Output(name="r", value=top3.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == [95.0, 92.0, 88.0]


class TestTagMerger:
    """Merge two tag-lists, deduplicate, and sort."""

    @pytest.mark.asyncio
    async def test_merge_tags(self):
        blog = ConstList(value=["python", "testing", "ci"])
        repo = ConstList(value=["python", "github", "ci", "automation"])
        combined = Extend(values=blog.output, other_values=repo.output)
        unique = Dedupe(values=combined.output)
        ordered = Sort(values=unique.output)
        sink = Output(name="r", value=ordered.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == ["automation", "ci", "github", "python", "testing"]


class TestSetTheory:
    """Set-operation nodes via graph."""

    @pytest.mark.asyncio
    async def test_shared_subscribers(self):
        a = ConstList(value=[101, 204, 305, 407])
        b = ConstList(value=[204, 305, 512, 618])
        shared = Intersection(list1=a.output, list2=b.output)
        ordered = Sort(values=shared.output)
        sink = Output(name="r", value=ordered.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == [204, 305]

    @pytest.mark.asyncio
    async def test_all_subscribers(self):
        a = ConstList(value=[1, 2, 3])
        b = ConstList(value=[3, 4, 5])
        merged = Union(list1=a.output, list2=b.output)
        ordered = Sort(values=merged.output)
        sink = Output(name="r", value=ordered.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_exclusive_to_first(self):
        a = ConstList(value=[10, 20, 30, 40])
        b = ConstList(value=[20, 40])
        only_a = Difference(list1=a.output, list2=b.output)
        ordered = Sort(values=only_a.output)
        sink = Output(name="r", value=ordered.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == [10, 30]


class TestChunkedProcessing:
    """Batch processing via Chunk node."""

    @pytest.mark.asyncio
    async def test_chunk_sizes(self):
        items = ConstList(value=list(range(1, 18)))
        batches = Chunk(values=items.output, chunk_size=5)
        n = ListLength(values=batches.output)
        sink = Output(name="n", value=n.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["n"] == 4  # 5+5+5+2


class TestListArithmetic:
    @pytest.mark.asyncio
    async def test_product_of_primes(self):
        primes = ConstList(value=[2.0, 3.0, 5.0, 7.0])
        result = Product(values=primes.output)
        sink = Output(name="r", value=result.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == 210.0

    @pytest.mark.asyncio
    async def test_range_then_select(self):
        seq = ListRange(start=100, stop=110, step=1)
        picked = SelectElements(values=seq.output, indices=[0, 5, 9])
        sink = Output(name="r", value=picked.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == [100, 105, 109]


class TestNestedFlatten:
    @pytest.mark.asyncio
    async def test_flatten_ragged(self):
        ragged = ConstList(value=[[1], [2, 3], [4, 5, 6]])
        flat = Flatten(values=ragged.output, max_depth=1)
        sink = Output(name="r", value=flat.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == [1, 2, 3, 4, 5, 6]

    @pytest.mark.asyncio
    async def test_flatten_deeply_nested(self):
        deep = ConstList(value=[[[["a"]], [["b"]]], [[["c"]]]])
        flat = Flatten(values=deep.output, max_depth=10)
        sink = Output(name="r", value=flat.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == ["a", "b", "c"]


# ======================================================================
# DICTIONARY PIPELINES
# ======================================================================


class TestUserProfileBuilder:
    """Build and update user profile dicts via graph."""

    @pytest.mark.asyncio
    async def test_create_from_parts(self):
        keys = ConstList(value=["username", "plan", "active"])
        vals = ConstList(value=["alice42", "pro", True])
        profile = Zip(keys=keys.output, values=vals.output)
        name = GetValue(dictionary=profile.output, key="username")
        sink = Output(name="r", value=name.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == "alice42"

    @pytest.mark.asyncio
    async def test_upgrade_plan(self):
        base = ConstDict(value={"username": "bob", "plan": "free", "credits": 5})
        overrides = ConstDict(
            value={"plan": "enterprise", "credits": 500}
        )
        upgraded = Update(dictionary=base.output, new_pairs=overrides.output)
        plan = GetValue(dictionary=upgraded.output, key="plan")
        sink = Output(name="r", value=plan.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == "enterprise"


class TestConfigMerger:
    """Overlay env-specific config on defaults."""

    @pytest.mark.asyncio
    async def test_dev_overrides_defaults(self):
        defaults = ConstDict(
            value={"db_host": "localhost", "db_port": 5432, "debug": False}
        )
        dev = ConstDict(value={"debug": True, "db_host": "dev-db.internal"})
        merged = Combine(dict_a=defaults.output, dict_b=dev.output)
        debug_val = GetValue(dictionary=merged.output, key="debug")
        host_val = GetValue(dictionary=merged.output, key="db_host")
        o1 = Output(name="debug", value=debug_val.output)
        o2 = Output(name="host", value=host_val.output)
        bag = await run_graph_async(create_graph(o1, o2))
        assert bag["debug"] is True
        assert bag["host"] == "dev-db.internal"


class TestSensitiveFieldRemover:
    @pytest.mark.asyncio
    async def test_strip_token(self):
        record = ConstDict(
            value={"id": 1, "name": "Alice", "api_token": "sk-secret"}
        )
        cleaned = Remove(dictionary=record.output, key="api_token")
        name = GetValue(dictionary=cleaned.output, key="name")
        sink = Output(name="r", value=name.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == "Alice"


class TestDictSubsetFilter:
    @pytest.mark.asyncio
    async def test_pick_public_fields(self):
        full = ConstDict(
            value={"name": "Bob", "email": "b@x.co", "ssn": "123-45-6789", "age": 30}
        )
        public = DictFilter(dictionary=full.output, keys=["name", "age"])
        sink = Output(name="r", value=public.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == {"name": "Bob", "age": 30}


class TestJSONRoundTrip:
    @pytest.mark.asyncio
    async def test_serialise_and_restore(self):
        original = ConstDict(value={"items": [1, 2, 3], "meta": {"v": 2}})
        as_str = ToJSON(dictionary=original.output)
        restored = ParseJSON(json_string=as_str.output)
        sink = Output(name="r", value=restored.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == {"items": [1, 2, 3], "meta": {"v": 2}}

    @pytest.mark.asyncio
    async def test_yaml_contains_keys(self):
        data = ConstDict(value={"service": "web", "replicas": 3})
        yml = ToYAML(dictionary=data.output)
        sink = Output(name="r", value=yml.output)
        bag = await run_graph_async(create_graph(sink))
        assert "service:" in bag["r"]
        assert "replicas:" in bag["r"]


class TestClassificationScorer:
    """ArgMax picks the highest-confidence label."""

    @pytest.mark.asyncio
    async def test_best_label(self):
        scores = ConstDict(value={"spam": 0.12, "ham": 0.85, "promo": 0.03})
        winner = ArgMax(scores=scores.output)
        sink = Output(name="r", value=winner.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == "ham"


# ======================================================================
# MIXED LIST + DICT PIPELINES
# ======================================================================


class TestLeaderboard:
    """Build a leaderboard from scores, identify the leader."""

    @pytest.mark.asyncio
    async def test_best_player(self):
        scores = ConstDict(
            value={"alice": 0.91, "bob": 0.78, "carol": 0.95, "dave": 0.83}
        )
        best = ArgMax(scores=scores.output)
        sink = Output(name="r", value=best.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["r"] == "carol"


class TestFeatureTogglePipeline:
    """Look up feature flags from a config dict."""

    @pytest.mark.asyncio
    async def test_toggle_lookup(self):
        toggles = ConstDict(
            value={"dark_mode": True, "beta_search": False, "ai_assist": True}
        )
        dm = GetValue(dictionary=toggles.output, key="dark_mode", default=False)
        bs = GetValue(dictionary=toggles.output, key="beta_search", default=False)
        missing = GetValue(
            dictionary=toggles.output, key="nonexistent", default="off"
        )
        o1 = Output(name="dm", value=dm.output)
        o2 = Output(name="bs", value=bs.output)
        o3 = Output(name="missing", value=missing.output)
        bag = await run_graph_async(create_graph(o1, o2, o3))
        assert bag["dm"] is True
        assert bag["bs"] is False
        assert bag["missing"] == "off"


class TestWarehouseReconciliation:
    """Compare warehouse stock lists using set operations, then summarise."""

    @pytest.mark.asyncio
    async def test_reconcile(self):
        wh_a = ConstList(value=["SKU-001", "SKU-002", "SKU-003", "SKU-005"])
        wh_b = ConstList(value=["SKU-002", "SKU-003", "SKU-004"])

        shared = Intersection(list1=wh_a.output, list2=wh_b.output)
        only_a = Difference(list1=wh_a.output, list2=wh_b.output)
        only_b = Difference(list1=wh_b.output, list2=wh_a.output)
        all_skus = Union(list1=wh_a.output, list2=wh_b.output)

        shared_sorted = Sort(values=shared.output)
        only_a_sorted = Sort(values=only_a.output)
        only_b_sorted = Sort(values=only_b.output)
        total_count = ListLength(values=all_skus.output)

        o1 = Output(name="shared", value=shared_sorted.output)
        o2 = Output(name="only_a", value=only_a_sorted.output)
        o3 = Output(name="only_b", value=only_b_sorted.output)
        o4 = Output(name="total", value=total_count.output)

        bag = await run_graph_async(create_graph(o1, o2, o3, o4))
        assert bag["shared"] == ["SKU-002", "SKU-003"]
        assert bag["only_a"] == ["SKU-001", "SKU-005"]
        assert bag["only_b"] == ["SKU-004"]
        assert bag["total"] == 5
