"""
Media Producer Campaign Builder DSL Example

Design a multi-asset content package for video and social channels using a
single creative brief. The workflow aligns scriptwriting, shot planning, and
social distribution deliverables for media producers.

Workflow:
1. **Campaign Inputs** - Capture the creative brief, target audience, and release timeline
2. **Narrative Script** - Generate a two-minute hero video script in the brand voice
3. **Shot List JSON** - Produce a structured shot list for the production crew
4. **Social Copy Pack** - Draft channel-specific posts to support distribution
5. **Release Plan** - Outline milestone-based rollout guidance for the campaign

This demonstrates:
- Coordinating multiple AI prompts off a single source of truth
- Converting JSON output from the LLM into a tabular shot list
- Packaging creative, production, and marketing deliverables together
- Supplying media producers with actionable, production-ready assets
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.metadata.types import LanguageModel, Provider
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.agents import Agent
from nodetool.dsl.nodetool.data import JSONToDataframe
from nodetool.dsl.nodetool.dictionary import MakeDictionary
from nodetool.dsl.nodetool.output import DictionaryOutput, DataframeOutput


# ---------------------------------------------------------------------------
# Campaign inputs (usually provided by the producer)
# ---------------------------------------------------------------------------
campaign_brief = StringInput(
    name="campaign_brief",
    description="Creative summary for the hero asset",
    value="""
Client: NovaWave Studios
Product: "Pulse" fitness tracker with adaptive coaching
Campaign Goal: Launch hero video and social teaser kit ahead of CES
Key Messages:
- Real-time coaching across cardio, strength, and recovery
- Works seamlessly across phone, watch, and connected equipment
- Designed for hybrid athletes balancing work and training
Tone & Visual Style: Energetic, cinematic, human-first
""",
)

target_audience = StringInput(
    name="target_audience",
    description="Primary audience insights",
    value="Hybrid professionals aged 25-40 who train before/after work, value data-driven feedback, and enjoy premium experiences.",
)

distribution_channels = StringInput(
    name="channels",
    description="Primary release channels",
    value="YouTube hero video, Instagram Reels, LinkedIn product announcement",
)

release_timing = StringInput(
    name="release_window",
    description="Timeline and milestones for the launch",
    value="CES preview week with public launch on January 3rd; paid amplification starting January 8th.",
)

brand_voice = StringInput(
    name="brand_voice",
    description="Tone/voice guidance",
    value="Confident, expert coach who celebrates progress and speaks with technical clarity without jargon.",
)

# Shared LLM configuration for creative generation
LLM = LanguageModel(
    type="language_model",
    id="gpt-4o",
    provider=Provider.OpenAI,
)


# ---------------------------------------------------------------------------
# Narrative script for the hero video
# ---------------------------------------------------------------------------
script_prompt = FormatText(
    template="""You are the lead creative director on a launch film.\n\nBrief:\n{{ brief }}\n\nAudience:\n{{ audience }}\n\nBrand Voice:\n{{ voice }}\n\nDeliver a two-minute hero video script with the following sections:\n1. Opening Hook (max 3 sentences)\n2. Product Narrative (3-4 beats with VO + suggested visuals)\n3. Proof Moments (2 customer scenarios)\n4. Closing Call-to-Action (tie back to hybrid athletes)\n\nFormat as Markdown with ### headings for each section and clearly label VO and Visual direction for every beat.",
    brief=campaign_brief.output,
    audience=target_audience.output,
    voice=brand_voice.output,
)

script_writer = Agent(
    prompt=script_prompt.output,
    model=LLM,
    system="You are an award-winning creative director delivering polished production scripts.",
    max_tokens=1200,
)


# ---------------------------------------------------------------------------
# Structured shot list in JSON
# ---------------------------------------------------------------------------
shot_list_prompt = FormatText(
    template="""Create a shot list for the hero video derived from this brief and audience context.\nReturn ONLY a JSON array where each object has keys: shot_number (int), setup (string), description (string), audio_direction (string), duration_seconds (int), and notes (string).\nInclude 8 shots that cover opening hook, product showcase, proof moments, and CTA.\nEnsure descriptions reflect the brand voice and tie back to the hybrid athlete lifestyle.\n\nBrief:\n{{ brief }}\n\nAudience:\n{{ audience }}\n\nChannels:\n{{ channels }}\n""",
    brief=campaign_brief.output,
    audience=target_audience.output,
    channels=distribution_channels.output,
)

shot_list_planner = Agent(
    prompt=shot_list_prompt.output,
    model=LLM,
    system="You are a meticulous line producer. Respond with valid JSON only, no backticks or commentary.",
    max_tokens=1100,
)

shot_list_table = JSONToDataframe(
    text=shot_list_planner.out.text,
)


# ---------------------------------------------------------------------------
# Social media copy pack aligned with hero film
# ---------------------------------------------------------------------------
social_copy_prompt = FormatText(
    template="""Draft social copy variants supporting the hero video.\nUse the campaign brief, audience, and release timing.\nProvide separate sections for YouTube description (120 words), Instagram Reel caption (150 characters, include two hashtags), and LinkedIn announcement (180 words with bullet list of key benefits).\nMaintain the brand voice.\n\nBrief:\n{{ brief }}\n\nAudience:\n{{ audience }}\n\nRelease Timing:\n{{ timing }}\n""",
    brief=campaign_brief.output,
    audience=target_audience.output,
    timing=release_timing.output,
)

social_copy_pack = Agent(
    prompt=social_copy_prompt.output,
    model=LLM,
    system="You craft channel-specific copy that is on-brand and launch-ready.",
    max_tokens=900,
)


# ---------------------------------------------------------------------------
# Release roadmap for producers and marketing
# ---------------------------------------------------------------------------
release_plan_prompt = FormatText(
    template="""Outline a production-to-launch roadmap for this media campaign.\nProvide milestones from pre-production through post-launch measurement.\nInclude columns: phase, owner, deliverables, due_date, and notes.\nDeliver the plan as Markdown table followed by a short list of risk mitigations.\n\nBrief:\n{{ brief }}\n\nRelease Window:\n{{ timing }}\n""",
    brief=campaign_brief.output,
    timing=release_timing.output,
)

release_plan = Agent(
    prompt=release_plan_prompt.output,
    model=LLM,
    system="You are an executive producer focused on timelines and accountability.",
    max_tokens=800,
)


# ---------------------------------------------------------------------------
# Bundle deliverables for downstream teams
# ---------------------------------------------------------------------------
creative_package = MakeDictionary(
    hero_video_script=script_writer.out.text,
    social_copy=social_copy_pack.out.text,
    release_plan=release_plan.out.text,
)

final_package = DictionaryOutput(
    name="media_campaign_package",
    description="Creative, social, and planning deliverables for the NovaWave Pulse launch.",
    value=creative_package.output,
)

shot_list_output = DataframeOutput(
    name="shot_list",
    description="Structured shot list for the production crew.",
    value=shot_list_table.output,
)


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
graph = create_graph(final_package, shot_list_output)


if __name__ == "__main__":
    result = run_graph(graph)
    print("🎬 Media campaign package generated!")
    print(result)
