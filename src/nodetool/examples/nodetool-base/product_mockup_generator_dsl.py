"""
Product Mockup Generator DSL Example

Produce coordinated product imagery for ecommerce listings in a single run.

Workflow:
1. **Brand Inputs** – Capture product, materials, brand tone, and backdrop direction.
2. **Prompt Assembly** – Build prompts for hero, lifestyle, and detail views.
3. **Text-to-Image Rendering** – Generate high-resolution assets tuned for each shot type.
4. **Prompt Reference Sheet** – Emit prompts for future regeneration or edits.
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.image import TextToImage
from nodetool.dsl.nodetool.output import ImageOutput, StringOutput
from nodetool.workflows.processing_context import AssetOutputMode


# --- Brand Inputs ------------------------------------------------------------
product_name = StringInput(
    name="product_name",
    description="Name and category of the product",
    value="Minimalist ceramic pour-over coffee brewer",
)

material_palette = StringInput(
    name="material_palette",
    description="Primary materials or finishes",
    value="Matte white ceramic with brushed brass accents",
)

brand_voice = StringInput(
    name="brand_voice",
    description="Desired brand tone or adjectives",
    value="Scandinavian minimalism, calm, premium craftsmanship",
)

backdrop_direction = StringInput(
    name="backdrop_direction",
    description="Background or scene direction",
    value="Soft morning light studio with neutral textures and subtle steam",
)

# --- Prompt Assembly ---------------------------------------------------------
base_brand_prompt = FormatText(
    template="""{{ product }} crafted from {{ materials }}. Brand tone: {{ voice }}. Render with crisp lighting, editorial clarity, realistic materials.""",
    product=product_name.output,
    materials=material_palette.output,
    voice=brand_voice.output,
)

hero_prompt = FormatText(
    template="""{{ base }} Shot type: hero product close-up on pedestal, symmetrical composition, 45-degree angle, clean shadows, negative space. Background: {{ backdrop }}.""",
    base=base_brand_prompt.output,
    backdrop=backdrop_direction.output,
)

lifestyle_prompt = FormatText(
    template="""{{ base }} Shot type: lifestyle scene on kitchen counter with curated props (fresh coffee beans, linen towel, ceramic cups). Capture steam and gentle sunlight, depth-of-field bokeh. Background: {{ backdrop }}.""",
    base=base_brand_prompt.output,
    backdrop=backdrop_direction.output,
)

detail_prompt = FormatText(
    template="""{{ base }} Shot type: macro detail focusing on pour spout texture, brass handle connection, and subtle surface reflections. Include droplets and tactile highlights.""",
    base=base_brand_prompt.output,
)

negative_prompt = (
    "low resolution, noisy texture, watermark, text overlay, extra limbs, distorted shapes, cluttered background, harsh shadows"
)

# --- Text-to-Image Rendering -------------------------------------------------
hero_render = TextToImage(
    prompt=hero_prompt.output,
    negative_prompt=negative_prompt,
    width=960,
    height=1280,
    guidance_scale=7.0,
    num_inference_steps=28,
)

lifestyle_render = TextToImage(
    prompt=lifestyle_prompt.output,
    negative_prompt=negative_prompt,
    width=1280,
    height=960,
    guidance_scale=6.5,
    num_inference_steps=26,
)

detail_render = TextToImage(
    prompt=detail_prompt.output,
    negative_prompt=negative_prompt,
    width=1024,
    height=1024,
    guidance_scale=7.2,
    num_inference_steps=30,
)

# --- Outputs -----------------------------------------------------------------
hero_output = ImageOutput(
    name="hero_product_shot",
    description="Primary ecommerce hero image",
    value=hero_render.output,
)

lifestyle_output = ImageOutput(
    name="lifestyle_scene",
    description="Styled environment shot",
    value=lifestyle_render.output,
)

detail_output = ImageOutput(
    name="detail_macro",
    description="Macro craftsmanship highlight",
    value=detail_render.output,
)

prompt_summary = FormatText(
    template="""# Product Mockup Prompts

- Hero Shot:\n{{ hero }}
- Lifestyle Scene:\n{{ lifestyle }}
- Detail Macro:\n{{ detail }}

Negative Prompt:\n{{ negative }}
""",
    hero=hero_prompt.output,
    lifestyle=lifestyle_prompt.output,
    detail=detail_prompt.output,
    negative=negative_prompt,
)

prompt_output = StringOutput(
    name="prompt_summary",
    description="Reference sheet for the generated prompts",
    value=prompt_summary.output,
)

# --- Graph -------------------------------------------------------------------
graph = create_graph(
    hero_output,
    lifestyle_output,
    detail_output,
    prompt_output,
)


if __name__ == "__main__":
    results = run_graph(graph, asset_output_mode=AssetOutputMode.WORKSPACE)
    print("Generated product visuals:")
    for key in ["hero_product_shot", "lifestyle_scene", "detail_macro"]:
        print(f"- {key}: {results[key]}")
    print("\nPrompts:\n", results["prompt_summary"])
