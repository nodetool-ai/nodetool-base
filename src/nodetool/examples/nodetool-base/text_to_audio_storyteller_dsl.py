"""
Bilingual Storyteller Text-to-Audio DSL Example

Create a narrated travel micro-adventure in two languages with distinct
voices, starting from a short creative brief.

Workflow:
1. **Capture Story Brief** - Provide the destination, hero, and tone
2. **Draft Narrative** - Use an LLM agent to write a 3-part travel story
3. **Translate Narrative** - Produce a Spanish rendition for bilingual delivery
4. **Generate Voices** - Convert each script to audio with different voice timbres
5. **Assemble Playlist** - Stitch the English and Spanish takes into a single track
6. **Expose Outputs** - Return both the text scripts and final audio asset

Highlights:
- Demonstrates OpenAI TextToSpeech DSL node with configurable model/voice
- Chains text generation, translation, and audio synthesis in one graph
- Produces multilingual deliverables ready for an audio tour or podcast
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.agents import Agent
from nodetool.dsl.openai.audio import TextToSpeech
from nodetool.dsl.nodetool.audio import ConcatList
from nodetool.dsl.nodetool.output import AudioOutput, StringOutput
from nodetool.metadata.types import LanguageModel, Provider


# Creative brief inputs
city_input = StringInput(
    name="destination_city",
    description="City featured in the travel story",
    value="Kyoto, Japan",
)

hero_input = StringInput(
    name="story_hero",
    description="Traveler or guide leading the experience",
    value="A curious sound designer named Aiko",
)

tone_input = StringInput(
    name="narration_tone",
    description="Desired vibe for the narration",
    value="warm, cinematic, and sensory-rich",
)


# Compose structured prompt for narrative generation
story_brief = FormatText(
    template="""You are an award-winning audio travel copywriter.
Craft a three-part narrated story for a micro audio tour.

Destination: {{ city }}
Guide: {{ hero }}
Tone guidance: {{ tone }}

Structure your markdown exactly as:
### Welcome & Hook
- ~60 words inviting listeners and previewing highlights

### Immersive Journey
- ~120 words covering three sensory-rich stops
- weave environmental sounds listeners should notice

### Invitation to Explore
- ~60 words closing with an inspiring call-to-action

Keep language vivid and voice-ready.
""",
    city=city_input.output,
    hero=hero_input.output,
    tone=tone_input.output,
)

story_writer = Agent(
    prompt=story_brief.output,
    model=LanguageModel(
        type="language_model",
        provider=Provider.OpenAI,
        id="gpt-4o-mini",
    ),
    system="You write immersive location-based audio scripts that sound great when spoken aloud.",
    max_tokens=750,
)


# Translate finished script for bilingual delivery
spanish_translation_prompt = FormatText(
    template="""You are a professional literary translator.
Render the following script into neutral Latin American Spanish while
preserving markdown headings and bullet points.

<ORIGINAL>
{{ script }}
</ORIGINAL>
""",
    script=story_writer.out.text,
)

spanish_version = Agent(
    prompt=spanish_translation_prompt.output,
    model=LanguageModel(
        type="language_model",
        provider=Provider.OpenAI,
        id="gpt-4o-mini",
    ),
    system="Translate creative writing with lyrical, natural Spanish phrasing suited for narration.",
    max_tokens=750,
)


# Convert both scripts to narrated audio using distinct voices
english_voice = TextToSpeech(
    input=story_writer.out.text,
    model=TextToSpeech.TtsModel.gpt_4o_mini_tts,
    voice=TextToSpeech.Voice.FABLE,
    speed=0.95,
)

spanish_voice = TextToSpeech(
    input=spanish_version.out.text,
    model=TextToSpeech.TtsModel.gpt_4o_mini_tts,
    voice=TextToSpeech.Voice.CORAL,
    speed=1.0,
)

# Merge takes into a single bilingual playlist
bilingual_mix = ConcatList(
    audio_files=[
        english_voice.output,
        spanish_voice.output,
    ],
)


# Expose outputs
story_output = StringOutput(
    name="english_story_markdown",
    value=story_writer.out.text,
)

spanish_output = StringOutput(
    name="spanish_story_markdown",
    value=spanish_version.out.text,
)

audio_output = AudioOutput(
    name="bilingual_story_audio",
    value=bilingual_mix.output,
)

# Build workflow graph with all deliverables
graph = create_graph(audio_output, story_output, spanish_output)


if __name__ == "__main__":
    result = run_graph(graph)
    print("🎧 Bilingual story ready!")
    print(f"Audio asset: {result['bilingual_story_audio']}")
    print("English script preview:\n", result["english_story_markdown"][:280], "...")
