"""
Collection-pipeline workflow tests.

Exercises list manipulation, dictionary transforms, and aggregation
nodes via direct process() invocation (list-typed inputs don't compose
well through the DSL graph runner's list-aggregation semantics).
"""

import json

import pytest

from nodetool.workflows.processing_context import ProcessingContext

from nodetool.nodes.nodetool.list import (
    Average,
    Chunk,
    Dedupe,
    Difference,
    Extend,
    Flatten,
    Intersection,
    ListRange,
    Maximum,
    Minimum,
    Product,
    Reverse,
    SelectElements,
    Slice,
    Sort,
    Sum as ListSum,
    Union,
)
from nodetool.nodes.nodetool.dictionary import (
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


@pytest.fixture
def ctx():
    return ProcessingContext(user_id="test", auth_token="test")


# ======================================================================
# LIST PIPELINES
# ======================================================================


class TestInventoryPipeline:
    """Simulate building and querying a product-count inventory."""

    @pytest.mark.asyncio
    async def test_build_inventory_counts(self, ctx):
        raw_counts = [4, 8, 2, 8, 4, 15, 2]
        unique = await Dedupe(values=raw_counts).process(ctx)
        ordered = await Sort(values=unique).process(ctx)
        assert ordered == [2, 4, 8, 15]

    @pytest.mark.asyncio
    async def test_restock_alert(self, ctx):
        stock = [0, 12, 3, 0, 7]
        total = await ListSum(values=[float(x) for x in stock]).process(ctx)
        low = await Minimum(values=[float(x) for x in stock]).process(ctx)
        assert total == 22.0
        assert low == 0.0


class TestGradePipeline:
    """Compute statistics for a classroom of grades."""

    @pytest.mark.asyncio
    async def test_grade_stats(self, ctx):
        grades = [78.0, 92.0, 65.0, 88.0, 95.0, 71.0]
        avg = await Average(values=grades).process(ctx)
        top = await Maximum(values=grades).process(ctx)
        bottom = await Minimum(values=grades).process(ctx)
        assert 80.0 < avg < 83.0  # exact is 81.5
        assert top == 95.0
        assert bottom == 65.0

    @pytest.mark.asyncio
    async def test_top_three_grades(self, ctx):
        grades = [78.0, 92.0, 65.0, 88.0, 95.0, 71.0]
        ordered = await Sort(values=grades).process(ctx)
        rev = await Reverse(values=ordered).process(ctx)
        top3 = await Slice(values=rev, start=0, stop=3).process(ctx)
        assert top3 == [95.0, 92.0, 88.0]


class TestTagMerger:
    """Merge two tag-lists, deduplicate, and sort alphabetically."""

    @pytest.mark.asyncio
    async def test_merge_tags(self, ctx):
        blog_tags = ["python", "testing", "ci"]
        repo_tags = ["python", "github", "ci", "automation"]
        combined = await Extend(values=blog_tags, other_values=repo_tags).process(ctx)
        unique = await Dedupe(values=combined).process(ctx)
        ordered = await Sort(values=unique).process(ctx)
        assert ordered == ["automation", "ci", "github", "python", "testing"]


class TestSetTheory:
    """Demonstrate set-operation nodes."""

    @pytest.mark.asyncio
    async def test_shared_subscribers(self, ctx):
        newsletter_a = [101, 204, 305, 407]
        newsletter_b = [204, 305, 512, 618]
        shared = await Intersection(
            list1=newsletter_a, list2=newsletter_b
        ).process(ctx)
        assert sorted(shared) == [204, 305]

    @pytest.mark.asyncio
    async def test_all_subscribers(self, ctx):
        a = [1, 2, 3]
        b = [3, 4, 5]
        all_subs = await Union(list1=a, list2=b).process(ctx)
        assert sorted(all_subs) == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_exclusive_to_first(self, ctx):
        a = [10, 20, 30, 40]
        b = [20, 40]
        only_a = await Difference(list1=a, list2=b).process(ctx)
        assert sorted(only_a) == [10, 30]


class TestChunkedProcessing:
    """Simulate batch processing via Chunk → per-batch operations."""

    @pytest.mark.asyncio
    async def test_chunk_sizes(self, ctx):
        items = list(range(1, 18))  # 1..17
        batches = await Chunk(values=items, chunk_size=5).process(ctx)
        assert len(batches) == 4  # 5+5+5+2
        assert batches[-1] == [16, 17]

    @pytest.mark.asyncio
    async def test_batch_sums(self, ctx):
        items = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        batches = await Chunk(values=items, chunk_size=3).process(ctx)
        # sum each batch
        sums = []
        for batch in batches:
            s = await ListSum(values=batch).process(ctx)
            sums.append(s)
        assert sums == [60.0, 150.0]


class TestListArithmetic:
    @pytest.mark.asyncio
    async def test_product_of_primes(self, ctx):
        primes = [2.0, 3.0, 5.0, 7.0]
        result = await Product(values=primes).process(ctx)
        assert result == 210.0

    @pytest.mark.asyncio
    async def test_range_then_select(self, ctx):
        seq = await ListRange(start=100, stop=110, step=1).process(ctx)
        picked = await SelectElements(values=seq, indices=[0, 5, 9]).process(ctx)
        assert picked == [100, 105, 109]


class TestNestedFlatten:
    @pytest.mark.asyncio
    async def test_flatten_ragged(self, ctx):
        ragged = [[1], [2, 3], [4, 5, 6]]
        flat = await Flatten(values=ragged, max_depth=1).process(ctx)
        assert flat == [1, 2, 3, 4, 5, 6]

    @pytest.mark.asyncio
    async def test_flatten_deeply_nested(self, ctx):
        deep = [[[["a"]], [["b"]]], [[["c"]]]]
        flat = await Flatten(values=deep, max_depth=10).process(ctx)
        assert flat == ["a", "b", "c"]


# ======================================================================
# DICTIONARY PIPELINES
# ======================================================================


class TestUserProfileBuilder:
    """Build, update, and serialise a user profile dict."""

    @pytest.mark.asyncio
    async def test_create_from_parts(self, ctx):
        profile = await Zip(
            keys=["username", "plan", "active"],
            values=["alice42", "pro", True],
        ).process(ctx)
        assert profile["username"] == "alice42"
        assert profile["active"] is True

    @pytest.mark.asyncio
    async def test_upgrade_plan(self, ctx):
        base = {"username": "bob", "plan": "free", "credits": 5}
        upgraded = await Update(
            dictionary=base,
            new_pairs={"plan": "enterprise", "credits": 500},
        ).process(ctx)
        assert upgraded["plan"] == "enterprise"
        assert upgraded["credits"] == 500
        assert upgraded["username"] == "bob"


class TestConfigMerger:
    """Overlay environment-specific config on top of defaults."""

    @pytest.mark.asyncio
    async def test_dev_overrides_defaults(self, ctx):
        defaults = {"db_host": "localhost", "db_port": 5432, "debug": False}
        dev = {"debug": True, "db_host": "dev-db.internal"}
        merged = await Combine(dict_a=defaults, dict_b=dev).process(ctx)
        assert merged["debug"] is True
        assert merged["db_host"] == "dev-db.internal"
        assert merged["db_port"] == 5432


class TestSensitiveFieldRemover:
    @pytest.mark.asyncio
    async def test_strip_token(self, ctx):
        record = {"id": 1, "name": "Alice", "api_token": "sk-secret"}
        cleaned = await Remove(dictionary=record, key="api_token").process(ctx)
        assert "api_token" not in cleaned
        assert cleaned["name"] == "Alice"


class TestDictSubsetFilter:
    @pytest.mark.asyncio
    async def test_pick_public_fields(self, ctx):
        full = {"name": "Bob", "email": "b@x.co", "ssn": "123-45-6789", "age": 30}
        public = await DictFilter(
            dictionary=full, keys=["name", "age"]
        ).process(ctx)
        assert public == {"name": "Bob", "age": 30}


class TestJSONRoundTrip:
    @pytest.mark.asyncio
    async def test_serialise_and_restore(self, ctx):
        original = {"items": [1, 2, 3], "meta": {"v": 2}}
        as_str = await ToJSON(dictionary=original).process(ctx)
        restored = await ParseJSON(json_string=as_str).process(ctx)
        assert restored == original

    @pytest.mark.asyncio
    async def test_yaml_contains_keys(self, ctx):
        data = {"service": "web", "replicas": 3}
        yml = await ToYAML(dictionary=data).process(ctx)
        assert "service:" in yml
        assert "replicas:" in yml


class TestClassificationScorer:
    """ArgMax picks the highest-confidence label."""

    @pytest.mark.asyncio
    async def test_best_label(self, ctx):
        scores = {"spam": 0.12, "ham": 0.85, "promo": 0.03}
        winner = await ArgMax(scores=scores).process(ctx)
        assert winner == "ham"


# ======================================================================
# MIXED LIST + DICT PIPELINES
# ======================================================================


class TestLeaderboard:
    """Build a leaderboard from scores, pick top-3, serialise."""

    @pytest.mark.asyncio
    async def test_top_scores(self, ctx):
        scores = {"alice": 0.91, "bob": 0.78, "carol": 0.95, "dave": 0.83}
        best_player = await ArgMax(scores=scores).process(ctx)
        assert best_player == "carol"

        # Get all scores as a sortable list
        score_vals = list(scores.values())
        ordered = await Sort(values=score_vals).process(ctx)
        top2 = await Slice(values=await Reverse(values=ordered).process(ctx),
                           start=0, stop=2).process(ctx)
        assert top2 == [0.95, 0.91]


class TestFeatureTogglePipeline:
    """Check feature flags from a config dict."""

    @pytest.mark.asyncio
    async def test_toggle_lookup(self, ctx):
        toggles = {"dark_mode": True, "beta_search": False, "ai_assist": True}
        dm = await GetValue(
            dictionary=toggles, key="dark_mode", default=False
        ).process(ctx)
        bs = await GetValue(
            dictionary=toggles, key="beta_search", default=False
        ).process(ctx)
        missing = await GetValue(
            dictionary=toggles, key="nonexistent", default="off"
        ).process(ctx)
        assert dm is True
        assert bs is False
        assert missing == "off"


class TestInventoryReconciliation:
    """Compare warehouse stock lists using set operations, then summarise."""

    @pytest.mark.asyncio
    async def test_reconcile(self, ctx):
        warehouse_a = ["SKU-001", "SKU-002", "SKU-003", "SKU-005"]
        warehouse_b = ["SKU-002", "SKU-003", "SKU-004"]

        shared_skus = await Intersection(
            list1=warehouse_a, list2=warehouse_b
        ).process(ctx)
        only_a = await Difference(list1=warehouse_a, list2=warehouse_b).process(ctx)
        only_b = await Difference(list1=warehouse_b, list2=warehouse_a).process(ctx)
        all_skus = await Union(list1=warehouse_a, list2=warehouse_b).process(ctx)

        assert sorted(shared_skus) == ["SKU-002", "SKU-003"]
        assert sorted(only_a) == ["SKU-001", "SKU-005"]
        assert sorted(only_b) == ["SKU-004"]
        assert len(all_skus) == 5

        summary = await Zip(
            keys=["shared", "only_a", "only_b", "total"],
            values=[len(shared_skus), len(only_a), len(only_b), len(all_skus)],
        ).process(ctx)
        summary_json = await ToJSON(dictionary=summary).process(ctx)
        parsed = json.loads(summary_json)
        assert parsed["total"] == 5
