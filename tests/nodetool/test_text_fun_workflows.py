"""
Fun text workflow integration tests.

Each test wires up a realistic multi-step text processing pipeline using
the DSL graph runner — the same way a user would compose nodes in the
visual editor, but with funnier names.
"""

import pytest

from nodetool.dsl.graph import create_graph, run_graph_async
from nodetool.dsl.nodetool.constant import String, Integer, Float  # noqa: F401
from nodetool.dsl.nodetool.text import (  # noqa: F401
    Concat,
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
# 1. Spaghetti Code Detangler
# ---------------------------------------------------------------------------
class TestSpaghettiCodeDetanglerWorkflow:
    """Trim + collapse whitespace + slugify to normalize messy variable names."""

    @pytest.mark.asyncio
    async def test_normalize_messy_variable_name(self):
        """Turn a horrifying variable name into a clean slug."""
        raw = String(value="   my   Terrible_VAR   name  ")
        step1 = TrimWhitespace(text=raw.output)
        step2 = CollapseWhitespace(text=step1.output)
        step3 = Slugify(text=step2.output)
        sink = Output(name="clean_var", value=step3.output)

        bag = await run_graph_async(create_graph(sink))
        assert " " not in bag["clean_var"]
        assert bag["clean_var"] == bag["clean_var"].lower()
        assert "my" in bag["clean_var"]


# ---------------------------------------------------------------------------
# 2. YOLO Password Checker
# ---------------------------------------------------------------------------
class TestYOLOPasswordCheckerWorkflow:
    """Check if a password has uppercase, lowercase, and numbers via regex."""

    @pytest.mark.asyncio
    async def test_strong_password_passes(self):
        """A decent password should pass all three regex checks."""
        pw = String(value="Hunter42isGreat")
        has_upper = RegexValidate(text=pw.output, pattern=r".*[A-Z].*")
        has_lower = RegexValidate(text=pw.output, pattern=r".*[a-z].*")
        has_digit = RegexValidate(text=pw.output, pattern=r".*\d.*")

        out1 = Output(name="has_upper", value=has_upper.output)
        out2 = Output(name="has_lower", value=has_lower.output)
        out3 = Output(name="has_digit", value=has_digit.output)

        bag = await run_graph_async(create_graph(out1, out2, out3))
        assert bag["has_upper"] is True
        assert bag["has_lower"] is True
        assert bag["has_digit"] is True

    @pytest.mark.asyncio
    async def test_weak_password_fails_digit_check(self):
        """A password without digits should fail that check."""
        pw = String(value="nopenope")
        has_digit = RegexValidate(text=pw.output, pattern=r".*\d.*")
        sink = Output(name="has_digit", value=has_digit.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["has_digit"] is False


# ---------------------------------------------------------------------------
# 3. Boomer To Zoomer Translator
# ---------------------------------------------------------------------------
class TestBoomerToZoomerTranslatorWorkflow:
    """Replace formal words with casual ones because language evolves, bro."""

    @pytest.mark.asyncio
    async def test_translate_formal_to_casual(self):
        """Replace stiff corporate speak with something more chill."""
        formal = String(value="That is excellent and I am very happy")
        step1 = Replace(text=formal.output, old="excellent", new="fire")
        step2 = Replace(text=step1.output, old="very happy", new="vibing")
        step3 = Replace(text=step2.output, old="That is", new="Thats")
        sink = Output(name="zoomer", value=step3.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["zoomer"] == "Thats fire and I am vibing"


# ---------------------------------------------------------------------------
# 4. Procrastinator Excuse Generator
# ---------------------------------------------------------------------------
class TestProcrastinatorExcuseGeneratorWorkflow:
    """Format text template with excuse parts — for the chronically late."""

    @pytest.mark.asyncio
    async def test_generate_excuse(self):
        """Fill in the excuse template with maximum plausibility."""
        reason = String(value="my cat sat on the keyboard")
        duration = String(value="3 hours")

        excuse = FormatText(
            template="Sorry I'm late, {{ reason }} and it took {{ duration }} to fix.",
            reason=reason.output,
            duration=duration.output,
        )
        sink = Output(name="excuse", value=excuse.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["excuse"] == "Sorry I'm late, my cat sat on the keyboard and it took 3 hours to fix."


# ---------------------------------------------------------------------------
# 5. Couch Potato Resume Builder
# ---------------------------------------------------------------------------
class TestCouchPotatoResumeBuilderWorkflow:
    """Concat multiple resume sections into one text — minimal effort edition."""

    @pytest.mark.asyncio
    async def test_build_resume(self):
        """Stitch together a resume from its lazy components."""
        header = String(value="Name: Lazy McSlacker")
        skills = String(value="Skills: Netflix, Napping")
        exp = String(value="Experience: 10 years of couch surfing")

        part1 = Concat(a=header.output, b=String(value="\n").output)
        part2 = Concat(a=part1.output, b=skills.output)
        part3 = Concat(a=part2.output, b=String(value="\n").output)
        full = Concat(a=part3.output, b=exp.output)
        sink = Output(name="resume", value=full.output)

        bag = await run_graph_async(create_graph(sink))
        assert "Lazy McSlacker" in bag["resume"]
        assert "Netflix" in bag["resume"]
        assert "couch surfing" in bag["resume"]


# ---------------------------------------------------------------------------
# 6. Overcaffeinated Editor
# ---------------------------------------------------------------------------
class TestOvercaffeinatedEditorWorkflow:
    """Check text length, starts/ends with, has content — triple-shot analysis."""

    @pytest.mark.asyncio
    async def test_analyze_article_opening(self):
        """Run multiple checks on a piece of text like an editor on their fifth espresso."""
        text = String(value="BREAKING: Coffee shortage hits downtown!")
        char_count = TextLength(text=text.output)
        starts_brk = StartsWith(text=text.output, prefix="BREAKING")
        ends_excl = EndsWith(text=text.output, suffix="!")
        has_coffee = Contains(text=text.output, substring="Coffee")

        out1 = Output(name="length", value=char_count.output)
        out2 = Output(name="is_breaking", value=starts_brk.output)
        out3 = Output(name="ends_bang", value=ends_excl.output)
        out4 = Output(name="has_coffee", value=has_coffee.output)

        bag = await run_graph_async(create_graph(out1, out2, out3, out4))
        assert bag["length"] == 40
        assert bag["is_breaking"] is True
        assert bag["ends_bang"] is True
        assert bag["has_coffee"] is True


# ---------------------------------------------------------------------------
# 7. Grammar Nazi Red Pen
# ---------------------------------------------------------------------------
class TestGrammarNaziRedPenWorkflow:
    """Regex replace common grammar mistakes — the internet needs this."""

    @pytest.mark.asyncio
    async def test_fix_common_mistakes(self):
        """Correct 'your/you're' and 'their/they're' with regex."""
        text = String(value="Your going to love this and there going too")
        step1 = RegexReplace(
            text=text.output,
            pattern=r"\bYour going\b",
            replacement="You're going",
        )
        step2 = RegexReplace(
            text=step1.output,
            pattern=r"\bthere going\b",
            replacement="they're going",
        )
        step3 = RegexReplace(
            text=step2.output,
            pattern=r"\btoo$",
            replacement="too!",
        )
        sink = Output(name="fixed", value=step3.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["fixed"] == "You're going to love this and they're going too!"


# ---------------------------------------------------------------------------
# 8. Midnight Snack Recipe Formatter
# ---------------------------------------------------------------------------
class TestMidnightSnackRecipeFormatterWorkflow:
    """Title case + surround with HTML tags for a recipe title."""

    @pytest.mark.asyncio
    async def test_format_recipe_title(self):
        """Make a sloppy recipe title presentable for the cookbook."""
        raw = String(value="leftover pizza quesadilla surprise")
        titled = ToTitlecase(text=raw.output)
        wrapped = SurroundWith(text=titled.output, prefix="<h1>", suffix="</h1>")
        sink = Output(name="title", value=wrapped.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["title"] == "<h1>Leftover Pizza Quesadilla Surprise</h1>"


# ---------------------------------------------------------------------------
# 9. Cat Name Generator Pipeline
# ---------------------------------------------------------------------------
class TestCatNameGeneratorPipelineWorkflow:
    """Format template to build silly cat names — the internet demands it."""

    @pytest.mark.asyncio
    async def test_generate_cat_name(self):
        """Assemble a majestic cat name from parts."""
        adj = String(value="Fluffy")
        noun = String(value="Whiskers")
        title = String(value="III")

        name = FormatText(
            template="Sir {{ adj }} {{ noun }} the {{ title }}",
            adj=adj.output,
            noun=noun.output,
            title=title.output,
        )
        sink = Output(name="cat_name", value=name.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["cat_name"] == "Sir Fluffy Whiskers the III"


# ---------------------------------------------------------------------------
# 10. Pizza Order Validator
# ---------------------------------------------------------------------------
class TestPizzaOrderValidatorWorkflow:
    """Contains checks for required pizza order fields — no pineapple allowed."""

    @pytest.mark.asyncio
    async def test_valid_order(self):
        """Verify a pizza order has all required fields."""
        order = String(value="Size: Large, Topping: Pepperoni, Crust: Thin")
        has_size = Contains(text=order.output, substring="Size:")
        has_topping = Contains(text=order.output, substring="Topping:")
        has_crust = Contains(text=order.output, substring="Crust:")
        has_pineapple = Contains(text=order.output, substring="Pineapple")

        out1 = Output(name="has_size", value=has_size.output)
        out2 = Output(name="has_topping", value=has_topping.output)
        out3 = Output(name="has_crust", value=has_crust.output)
        out4 = Output(name="no_pineapple", value=has_pineapple.output)

        bag = await run_graph_async(create_graph(out1, out2, out3, out4))
        assert bag["has_size"] is True
        assert bag["has_topping"] is True
        assert bag["has_crust"] is True
        assert bag["no_pineapple"] is False  # as it should be


# ---------------------------------------------------------------------------
# 11. Empty Fridge Detector
# ---------------------------------------------------------------------------
class TestEmptyFridgeDetectorWorkflow:
    """Trim + is empty check for a shopping list — time to order takeout."""

    @pytest.mark.asyncio
    async def test_empty_list_detected(self):
        """A whitespace-only shopping list means the fridge is empty."""
        shopping_list = String(value="   \t  \n  ")
        trimmed = TrimWhitespace(text=shopping_list.output)
        empty = IsEmpty(text=trimmed.output)
        sink = Output(name="fridge_empty", value=empty.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["fridge_empty"] is True

    @pytest.mark.asyncio
    async def test_non_empty_list(self):
        """A list with actual items means we're good... for now."""
        shopping_list = String(value="  milk, eggs  ")
        trimmed = TrimWhitespace(text=shopping_list.output)
        empty = IsEmpty(text=trimmed.output)
        sink = Output(name="fridge_empty", value=empty.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["fridge_empty"] is False


# ---------------------------------------------------------------------------
# 12. Shakespeare To Emoji
# ---------------------------------------------------------------------------
class TestShakespeareToEmojiWorkflow:
    """Replace Shakespearean words with text representations of emojis."""

    @pytest.mark.asyncio
    async def test_translate_bard_to_emoji_text(self):
        """Translate the Bard's finest into modern expression."""
        bard = String(value="To love or not to love, that is the question")
        step1 = Replace(text=bard.output, old="love", new="<3")
        step2 = Replace(text=step1.output, old="question", new="???")
        sink = Output(name="modern", value=step2.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["modern"] == "To <3 or not to <3, that is the ???"


# ---------------------------------------------------------------------------
# 13. Dad Joke Assembler
# ---------------------------------------------------------------------------
class TestDadJokeAssemblerWorkflow:
    """Concat parts of a dad joke — groan included free of charge."""

    @pytest.mark.asyncio
    async def test_assemble_dad_joke(self):
        """Build the ultimate dad joke from its component parts."""
        setup = String(value="Why don't scientists trust atoms? ")
        punchline = String(value="Because they make up everything!")
        joke = Concat(a=setup.output, b=punchline.output)
        sink = Output(name="joke", value=joke.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["joke"] == "Why don't scientists trust atoms? Because they make up everything!"


# ---------------------------------------------------------------------------
# 14. Sleep Deprived Email Fixer
# ---------------------------------------------------------------------------
class TestSleepDeprivedEmailFixerWorkflow:
    """Trim + collapse whitespace + capitalize — fix emails written at 3am."""

    @pytest.mark.asyncio
    async def test_fix_sleepy_email(self):
        """Clean up the mess you typed while half-asleep."""
        raw = String(value="   hi   bob   i   need   the   report   ")
        step1 = TrimWhitespace(text=raw.output)
        step2 = CollapseWhitespace(text=step1.output)
        step3 = CapitalizeText(text=step2.output)
        sink = Output(name="email", value=step3.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["email"] == "Hi bob i need the report"


# ---------------------------------------------------------------------------
# 15. Unicorn Startup Name Generator
# ---------------------------------------------------------------------------
class TestUnicornStartupNameGeneratorWorkflow:
    """Format + slugify for a startup URL — disrupting the slug industry."""

    @pytest.mark.asyncio
    async def test_generate_startup_slug(self):
        """Create a startup name and turn it into a URL-friendly slug."""
        prefix = String(value="Cloud")
        suffix = String(value="Synergy")
        name = FormatText(
            template="{{ prefix }} {{ suffix }} AI Platform",
            prefix=prefix.output,
            suffix=suffix.output,
        )
        slug = Slugify(text=name.output)
        sink = Output(name="url_slug", value=slug.output)

        bag = await run_graph_async(create_graph(sink))
        assert " " not in bag["url_slug"]
        assert "cloud" in bag["url_slug"]
        assert "synergy" in bag["url_slug"]


# ---------------------------------------------------------------------------
# 16. Dramatic Reading Preparer
# ---------------------------------------------------------------------------
class TestDramaticReadingPreparerWorkflow:
    """Uppercase + surround with stage directions — for maximum drama."""

    @pytest.mark.asyncio
    async def test_prepare_dramatic_text(self):
        """Prepare text for a dramatic reading at the local improv night."""
        line = String(value="to be or not to be")
        shouted = ToUppercase(text=line.output)
        staged = SurroundWith(
            text=shouted.output,
            prefix="[DRAMATICALLY] ",
            suffix=" [EXITS STAGE LEFT]",
        )
        sink = Output(name="drama", value=staged.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["drama"] == "[DRAMATICALLY] TO BE OR NOT TO BE [EXITS STAGE LEFT]"


# ---------------------------------------------------------------------------
# 17. Bug Report Sanitizer
# ---------------------------------------------------------------------------
class TestBugReportSanitizerWorkflow:
    """Regex replace sensitive info + trim — because logs have secrets."""

    @pytest.mark.asyncio
    async def test_sanitize_bug_report(self):
        """Strip API keys and emails from a bug report before sharing."""
        report = String(
            value="  Error with key=sk-abc123def456 from user@company.com  "
        )
        step1 = TrimWhitespace(text=report.output)
        step2 = RegexReplace(
            text=step1.output,
            pattern=r"key=\S+",
            replacement="key=[REDACTED]",
        )
        step3 = RegexReplace(
            text=step2.output,
            pattern=r"\S+@\S+\.\S+",
            replacement="[EMAIL_REDACTED]",
        )
        sink = Output(name="safe_report", value=step3.output)

        bag = await run_graph_async(create_graph(sink))
        assert "sk-abc123def456" not in bag["safe_report"]
        assert "user@company.com" not in bag["safe_report"]
        assert "[REDACTED]" in bag["safe_report"]
        assert "[EMAIL_REDACTED]" in bag["safe_report"]


# ---------------------------------------------------------------------------
# 18. Hashtag Grinder
# ---------------------------------------------------------------------------
class TestHashtagGrinderWorkflow:
    """Remove punctuation + lowercase + replace spaces — hashtag everything."""

    @pytest.mark.asyncio
    async def test_grind_hashtags(self):
        """Turn a phrase into a social media hashtag."""
        phrase = String(value="I Love Python!!!")
        step1 = RemovePunctuation(text=phrase.output)
        step2 = ToLowercase(text=step1.output)
        step3 = CollapseWhitespace(text=step2.output)
        step4 = TrimWhitespace(text=step3.output)
        step5 = Replace(text=step4.output, old=" ", new="")
        step6 = SurroundWith(text=step5.output, prefix="#", suffix="")
        sink = Output(name="hashtag", value=step6.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["hashtag"] == "#ilovepython"


# ---------------------------------------------------------------------------
# 19. Karen Complaint Redactor
# ---------------------------------------------------------------------------
class TestKarenComplaintRedactorWorkflow:
    """Regex replace names + phone numbers — protect the manager."""

    @pytest.mark.asyncio
    async def test_redact_complaint(self):
        """Redact personal info from an angry customer complaint."""
        complaint = String(
            value="I, Karen Smith, demand to speak to the manager! Call me at 555-123-4567!"
        )
        step1 = RegexReplace(
            text=complaint.output,
            pattern=r"Karen \w+",
            replacement="[NAME REDACTED]",
        )
        step2 = RegexReplace(
            text=step1.output,
            pattern=r"\d{3}-\d{3}-\d{4}",
            replacement="[PHONE REDACTED]",
        )
        sink = Output(name="redacted", value=step2.output)

        bag = await run_graph_async(create_graph(sink))
        assert "Karen Smith" not in bag["redacted"]
        assert "555-123-4567" not in bag["redacted"]
        assert "[NAME REDACTED]" in bag["redacted"]
        assert "[PHONE REDACTED]" in bag["redacted"]


# ---------------------------------------------------------------------------
# 20. Fortune Cookie Assembler
# ---------------------------------------------------------------------------
class TestFortuneCookieAssemblerWorkflow:
    """Format template with wisdom parts — lucky numbers not included."""

    @pytest.mark.asyncio
    async def test_assemble_fortune(self):
        """Bake a fortune cookie message from template parts."""
        wisdom = String(value="patience")
        object_ = String(value="rubber duck")
        fortune = FormatText(
            template="A great {{ wisdom }} begins with a single {{ object }}.",
            wisdom=wisdom.output,
            object=object_.output,
        )
        sink = Output(name="fortune", value=fortune.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["fortune"] == "A great patience begins with a single rubber duck."


# ---------------------------------------------------------------------------
# 21. Passive Aggressive Note Writer
# ---------------------------------------------------------------------------
class TestPassiveAggressiveNoteWriterWorkflow:
    """Format template + surround with stars — office warfare, text edition."""

    @pytest.mark.asyncio
    async def test_write_passive_aggressive_note(self):
        """Craft the perfect passive-aggressive office note."""
        name = String(value="Dave")
        item = String(value="my clearly labeled yogurt")
        note = FormatText(
            template="Dear {{ name }}, please stop eating {{ item }}. Thanks.",
            name=name.output,
            item=item.output,
        )
        decorated = SurroundWith(text=note.output, prefix="*** ", suffix=" ***")
        sink = Output(name="note", value=decorated.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["note"] == "*** Dear Dave, please stop eating my clearly labeled yogurt. Thanks. ***"


# ---------------------------------------------------------------------------
# 22. Ancient Scroll Decoder
# ---------------------------------------------------------------------------
class TestAncientScrollDecoderWorkflow:
    """HTML to text + trim + title case — unearthing digital artifacts."""

    @pytest.mark.asyncio
    async def test_decode_ancient_html(self):
        """Decode an ancient HTML scroll into readable titled text."""
        scroll = String(
            value="<div><p>the lost art of readable code</p></div>"
        )
        plain = HtmlToText(html=scroll.output)
        trimmed = TrimWhitespace(text=plain.output)
        titled = ToTitlecase(text=trimmed.output)
        sink = Output(name="decoded", value=titled.output)

        bag = await run_graph_async(create_graph(sink))
        assert "<div>" not in bag["decoded"]
        assert "<p>" not in bag["decoded"]
        assert bag["decoded"] == "The Lost Art Of Readable Code"


# ---------------------------------------------------------------------------
# 23. Cryptic Error Message Translator
# ---------------------------------------------------------------------------
class TestCrypticErrorMessageTranslatorWorkflow:
    """Replace technical jargon with plain English — for humans."""

    @pytest.mark.asyncio
    async def test_translate_error_message(self):
        """Turn a cryptic error into something your mom could understand."""
        error = String(value="SEGFAULT in NULL pointer dereference at 0xDEADBEEF")
        step1 = Replace(text=error.output, old="SEGFAULT", new="Crash")
        step2 = Replace(text=step1.output, old="NULL pointer dereference", new="something was missing")
        step3 = Replace(text=step2.output, old="at 0xDEADBEEF", new="somewhere in the code")
        sink = Output(name="translated", value=step3.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["translated"] == "Crash in something was missing somewhere in the code"


# ---------------------------------------------------------------------------
# 24. Monkey Typing Shakespeare
# ---------------------------------------------------------------------------
class TestMonkeyTypingShakespeareWorkflow:
    """Regex validate if text matches a pattern — infinite monkeys edition."""

    @pytest.mark.asyncio
    async def test_monkey_typed_hamlet(self):
        """Check if the monkey actually typed something Shakespearean."""
        text = String(value="To be or not to be")
        matches = RegexValidate(
            text=text.output,
            pattern=r"^To be or not to be$",
        )
        sink = Output(name="is_shakespeare", value=matches.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["is_shakespeare"] is True

    @pytest.mark.asyncio
    async def test_monkey_typed_gibberish(self):
        """The monkey just mashed the keyboard — not Shakespeare."""
        text = String(value="asdfjkl;qwerty")
        matches = RegexValidate(
            text=text.output,
            pattern=r"^To be or not to be$",
        )
        sink = Output(name="is_shakespeare", value=matches.output)

        bag = await run_graph_async(create_graph(sink))
        assert bag["is_shakespeare"] is False


# ---------------------------------------------------------------------------
# 25. Robot Poetry Formatter
# ---------------------------------------------------------------------------
class TestRobotPoetryFormatterWorkflow:
    """Split text + check line count + concat with newlines — beep boop verse."""

    @pytest.mark.asyncio
    async def test_format_robot_poem(self):
        """Split a flat poem into lines, verify structure, and reassemble."""
        flat_poem = String(value="beep|boop|whirr|click")
        lines = Split(text=flat_poem.output, delimiter="|")
        out_lines = Output(name="lines", value=lines.output)

        rejoined_text = Replace(text=flat_poem.output, old="|", new="\n")
        out_poem = Output(name="poem", value=rejoined_text.output)

        has_beep = Contains(text=flat_poem.output, substring="beep")
        out_has_beep = Output(name="has_beep", value=has_beep.output)

        bag = await run_graph_async(create_graph(out_lines, out_poem, out_has_beep))
        assert len(bag["lines"]) == 4
        assert bag["poem"] == "beep\nboop\nwhirr\nclick"
        assert bag["has_beep"] is True
