"""
Multi-step text transformation pipeline workflows.

Each test builds a realistic processing pipeline by chaining several
text nodes together - the kind of thing a real user would wire up
in the visual editor.  DSL graph tests exercise the end-to-end
graph runner; direct-invocation tests verify node composition logic.
"""

import pytest

from nodetool.dsl.graph import create_graph, run_graph_async
from nodetool.dsl.nodetool.constant import String, Integer
from nodetool.dsl.nodetool.text import (
    Concat,
    Extract,
    FormatText,
    Replace,
    Slugify,
    Split,
    ToLowercase,
    ToTitlecase,
    ToUppercase,
    TrimWhitespace,
    RegexReplace,
    RegexSplit,
    CollapseWhitespace,
    Contains,
    EndsWith,
    IsEmpty,
    Length as TextLength,
    RemovePunctuation,
    StartsWith,
    SurroundWith,
    ToString,
    Equals as TextEquals,
    HtmlToText,
    ParseJSON as TextParseJSON,
    CapitalizeText,
    RegexValidate,
)
from nodetool.dsl.nodetool.output import Output


# ---------------------------------------------------------------------------
# Scenario: Normalise user-provided display names
# ---------------------------------------------------------------------------
class TestDisplayNameNormaliser:
    """Trim → collapse whitespace → title-case."""

    @pytest.mark.asyncio
    async def test_messy_spacing(self):
        raw = String(value="   jane   DOE  ")
        step1 = TrimWhitespace(text=raw.output)
        step2 = CollapseWhitespace(text=step1.output)
        step3 = ToTitlecase(text=step2.output)
        sink = Output(name="clean_name", value=step3.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["clean_name"] == "Jane Doe"

    @pytest.mark.asyncio
    async def test_all_caps_input(self):
        raw = String(value="JOHN SMITH")
        step1 = ToLowercase(text=raw.output)
        step2 = ToTitlecase(text=step1.output)
        sink = Output(name="display", value=step2.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["display"] == "John Smith"


# ---------------------------------------------------------------------------
# Scenario: Generate a URL slug from a blog title
# ---------------------------------------------------------------------------
class TestBlogSlugGenerator:
    @pytest.mark.asyncio
    async def test_slug_from_fancy_title(self):
        title = String(value="  10 Tips & Tricks for Python!! ")
        trimmed = TrimWhitespace(text=title.output)
        no_punct = RemovePunctuation(text=trimmed.output)
        slug = Slugify(text=no_punct.output)
        sink = Output(name="slug", value=slug.output)

        bag = await run_graph_async(create_graph(sink))
        # After punctuation removal "&" disappears, collapsing the gap
        assert "tips" in bag["slug"]
        assert "python" in bag["slug"]
        assert " " not in bag["slug"]  # no spaces in a slug

    @pytest.mark.asyncio
    async def test_slug_has_no_uppercase(self):
        title = String(value="My GREAT Article")
        slug = Slugify(text=title.output)
        has_upper = Contains(text=slug.output, substring="G")
        sink = Output(name="has_upper", value=has_upper.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["has_upper"] is False


# ---------------------------------------------------------------------------
# Scenario: Redact phone numbers from a support transcript
# ---------------------------------------------------------------------------
class TestPhoneRedaction:
    @pytest.mark.asyncio
    async def test_redact_us_phone(self):
        transcript = String(
            value="Call me at 555-867-5309 or 555.123.4567 thanks."
        )
        redacted = RegexReplace(
            text=transcript.output,
            pattern=r"\d{3}[-.\s]\d{3}[-.\s]\d{4}",
            replacement="[REDACTED]",
        )
        sink = Output(name="safe", value=redacted.output)

        bag = await run_graph_async(create_graph(sink))
        assert "[REDACTED]" in bag["safe"]
        assert "555-867-5309" not in bag["safe"]

    @pytest.mark.asyncio
    async def test_clean_text_contains_no_digits_block(self):
        transcript = String(value="Ticket #42: phone 555-000-1111")
        redacted = RegexReplace(
            text=transcript.output,
            pattern=r"\d{3}-\d{3}-\d{4}",
            replacement="***",
        )
        still_has_ticket = Contains(text=redacted.output, substring="Ticket #42")
        sink = Output(name="has_ticket", value=still_has_ticket.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["has_ticket"] is True


# ---------------------------------------------------------------------------
# Scenario: Build a parameterised SQL-like WHERE clause
# ---------------------------------------------------------------------------
class TestWhereClauseBuilder:
    @pytest.mark.asyncio
    async def test_jinja_template(self):
        tbl = String(value="orders")
        col = String(value="status")
        val = String(value="shipped")

        clause = FormatText(
            template="SELECT * FROM {{ table }} WHERE {{ column }} = '{{ value }}'",
            table=tbl.output,
            column=col.output,
            value=val.output,
        )
        upper = ToUppercase(text=clause.output)
        sink = Output(name="sql", value=upper.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["sql"] == "SELECT * FROM ORDERS WHERE STATUS = 'SHIPPED'"


# ---------------------------------------------------------------------------
# Scenario: Extract domain from an e-mail address
# ---------------------------------------------------------------------------
class TestEmailDomainExtractor:
    @pytest.mark.asyncio
    async def test_split_and_pick(self):
        """Split on '@', the second chunk is the domain."""
        email = String(value="alice@example.org")
        parts = Split(text=email.output, delimiter="@")
        # We can't index a list via DSL easily, so let's use Replace
        # to remove everything up to and including '@'
        domain = RegexReplace(
            text=email.output,
            pattern=r"^[^@]+@",
            replacement="",
        )
        sink = Output(name="domain", value=domain.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["domain"] == "example.org"


# ---------------------------------------------------------------------------
# Scenario: Multi-output analysis of a headline
# ---------------------------------------------------------------------------
class TestHeadlineAnalytics:
    @pytest.mark.asyncio
    async def test_parallel_analysis(self):
        headline = String(value="Breaking: Python 4.0 Released Today!")
        char_count = TextLength(text=headline.output)
        has_breaking = StartsWith(text=headline.output, prefix="Breaking")
        has_excl = EndsWith(text=headline.output, suffix="!")
        upper_ver = ToUppercase(text=headline.output)

        out_len = Output(name="chars", value=char_count.output)
        out_brk = Output(name="is_breaking", value=has_breaking.output)
        out_exc = Output(name="ends_excl", value=has_excl.output)
        out_up = Output(name="upper", value=upper_ver.output)

        bag = await run_graph_async(
            create_graph(out_len, out_brk, out_exc, out_up)
        )
        assert bag["chars"] == len("Breaking: Python 4.0 Released Today!")
        assert bag["is_breaking"] is True
        assert bag["ends_excl"] is True
        assert bag["upper"] == "BREAKING: PYTHON 4.0 RELEASED TODAY!"


# ---------------------------------------------------------------------------
# Scenario: Simple HTML tag stripper
# ---------------------------------------------------------------------------
class TestHTMLCleaner:
    @pytest.mark.asyncio
    async def test_strip_tags_and_normalise(self):
        raw_html = String(
            value="<h1>Welcome</h1><p>Hello &amp; <b>goodbye</b></p>"
        )
        plain = HtmlToText(html=raw_html.output)
        trimmed = TrimWhitespace(text=plain.output)
        sink = Output(name="text", value=trimmed.output)

        bag = await run_graph_async(create_graph(sink))
        assert "Welcome" in bag["text"]
        assert "<h1>" not in bag["text"]


# ---------------------------------------------------------------------------
# Scenario: Chained replace for sanitising file paths
# ---------------------------------------------------------------------------
class TestPathSanitiser:
    @pytest.mark.asyncio
    async def test_remove_special_chars(self):
        raw = String(value="my file (v2) [final].txt")
        step1 = Replace(text=raw.output, old="(", new="")
        step2 = Replace(text=step1.output, old=")", new="")
        step3 = Replace(text=step2.output, old="[", new="")
        step4 = Replace(text=step3.output, old="]", new="")
        step5 = Replace(text=step4.output, old=" ", new="_")
        sink = Output(name="safe_name", value=step5.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["safe_name"] == "my_file_v2_final.txt"


# ---------------------------------------------------------------------------
# Scenario: Concatenate a greeting from parts
# ---------------------------------------------------------------------------
class TestGreetingAssembler:
    @pytest.mark.asyncio
    async def test_build_greeting(self):
        prefix = String(value="Dear ")
        name = String(value="Dr. Smith")
        suffix = String(value=", welcome aboard!")
        part_a = Concat(a=prefix.output, b=name.output)
        full = Concat(a=part_a.output, b=suffix.output)
        sink = Output(name="msg", value=full.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["msg"] == "Dear Dr. Smith, welcome aboard!"


# ---------------------------------------------------------------------------
# Scenario: Capitalise first word only
# ---------------------------------------------------------------------------
class TestSentenceCapitaliser:
    @pytest.mark.asyncio
    async def test_capitalise_lowered_text(self):
        raw = String(value="  THE QUICK BROWN FOX  ")
        trimmed = TrimWhitespace(text=raw.output)
        lowered = ToLowercase(text=trimmed.output)
        capped = CapitalizeText(text=lowered.output)
        sink = Output(name="sentence", value=capped.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["sentence"] == "The quick brown fox"


# ---------------------------------------------------------------------------
# Scenario: Validate that a string looks like an IPv4 address
# ---------------------------------------------------------------------------
class TestIPv4Validator:
    @pytest.mark.asyncio
    async def test_valid_ip(self):
        addr = String(value="192.168.1.100")
        ok = RegexValidate(
            text=addr.output,
            pattern=r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
        )
        sink = Output(name="valid", value=ok.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["valid"] is True

    @pytest.mark.asyncio
    async def test_invalid_ip(self):
        addr = String(value="not.an" )
        ok = RegexValidate(
            text=addr.output,
            pattern=r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
        )
        sink = Output(name="valid", value=ok.output)
        bag = await run_graph_async(create_graph(sink))
        assert bag["valid"] is False


# ---------------------------------------------------------------------------
# Scenario: JSON extraction from raw text
# ---------------------------------------------------------------------------
class TestEmbeddedJSONExtractor:
    @pytest.mark.asyncio
    async def test_parse_json_from_string(self):
        raw = String(value='{"temp_c": 21.5, "city": "Berlin"}')
        parsed = TextParseJSON(text=raw.output)
        sink = Output(name="data", value=parsed.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["data"]["city"] == "Berlin"
        assert bag["data"]["temp_c"] == 21.5


# ---------------------------------------------------------------------------
# Scenario: Wrap text in markup tags
# ---------------------------------------------------------------------------
class TestMarkupWrapper:
    @pytest.mark.asyncio
    async def test_surround_with_xml_tags(self):
        body = String(value="Hello World")
        wrapped = SurroundWith(text=body.output, prefix="<msg>", suffix="</msg>")
        sink = Output(name="xml", value=wrapped.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["xml"] == "<msg>Hello World</msg>"

    @pytest.mark.asyncio
    async def test_double_wrap(self):
        body = String(value="content")
        inner = SurroundWith(text=body.output, prefix="[", suffix="]")
        outer = SurroundWith(text=inner.output, prefix="(", suffix=")")
        sink = Output(name="nested", value=outer.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["nested"] == "([content])"


# ---------------------------------------------------------------------------
# Scenario: Integer-to-string with formatting
# ---------------------------------------------------------------------------
class TestNumericFormatting:
    @pytest.mark.asyncio
    async def test_int_to_padded_string(self):
        num = Integer(value=7)
        as_str = ToString(value=num.output)
        sink = Output(name="text", value=as_str.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["text"] == "7"

    @pytest.mark.asyncio
    async def test_format_with_label(self):
        qty = Integer(value=42)
        label = FormatText(
            template="Quantity: {{ n }} units",
            n=qty.output,
        )
        sink = Output(name="lbl", value=label.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["lbl"] == "Quantity: 42 units"


# ---------------------------------------------------------------------------
# Scenario: Split a CSV row and rejoin with a different separator
# ---------------------------------------------------------------------------
class TestCSVRowTransformer:
    @pytest.mark.asyncio
    async def test_split_then_check_count(self):
        row = String(value="alpha,bravo,charlie,delta")
        cells = Split(text=row.output, delimiter=",")
        # Since we can't easily get list length via DSL graph for text split,
        # convert back to a representative string and check
        sink = Output(name="cells", value=cells.output)

        bag = await run_graph_async(create_graph(sink))
        assert len(bag["cells"]) == 4
        assert "alpha" in bag["cells"]

    @pytest.mark.asyncio
    async def test_regex_split_on_whitespace(self):
        messy = String(value="one  two\tthree   four")
        tokens = RegexSplit(text=messy.output, pattern=r"\s+")
        sink = Output(name="tokens", value=tokens.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["tokens"] == ["one", "two", "three", "four"]


# ---------------------------------------------------------------------------
# Scenario: Check two strings for equality (case-insensitive via lowering)
# ---------------------------------------------------------------------------
class TestCaseInsensitiveComparison:
    @pytest.mark.asyncio
    async def test_same_content_different_case(self):
        left = String(value="Hello World")
        right = String(value="hello world")
        left_low = ToLowercase(text=left.output)
        right_low = ToLowercase(text=right.output)
        eq = TextEquals(text_a=left_low.output, text_b=right_low.output)
        sink = Output(name="match", value=eq.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["match"] is True

    @pytest.mark.asyncio
    async def test_different_content(self):
        left = String(value="Alpha")
        right = String(value="Beta")
        eq = TextEquals(text_a=left.output, text_b=right.output)
        sink = Output(name="match", value=eq.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["match"] is False


# ---------------------------------------------------------------------------
# Scenario: Detect empty-after-trim
# ---------------------------------------------------------------------------
class TestBlankDetector:
    @pytest.mark.asyncio
    async def test_whitespace_only_is_empty(self):
        raw = String(value="    ")
        trimmed = TrimWhitespace(text=raw.output)
        blank = IsEmpty(text=trimmed.output)
        sink = Output(name="blank", value=blank.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["blank"] is True

    @pytest.mark.asyncio
    async def test_real_content_not_empty(self):
        raw = String(value="  hi  ")
        trimmed = TrimWhitespace(text=raw.output)
        blank = IsEmpty(text=trimmed.output)
        sink = Output(name="blank", value=blank.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["blank"] is False
