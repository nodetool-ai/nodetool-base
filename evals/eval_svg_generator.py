"""
Evaluation script for the SVGGenerator node.

Tests the SVGGenerator node for generating SVG graphics from prompts.

Usage:
    python eval_svg_generator.py
    python eval_svg_generator.py --models ollama/gemma3:4b
    python eval_svg_generator.py --output results.json
"""

import time
from typing import List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from nodetool.metadata.types import LanguageModel
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.nodetool.generators import SVGGenerator

try:
    from .common import EvalCase as BaseEvalCase, EvalResult, EvalRunner, run_evaluation, get_provider
except ImportError:
    from common import EvalCase as BaseEvalCase, EvalResult, EvalRunner, run_evaluation, get_provider


class SVGTask(Enum):
    """Types of SVG generation tasks."""
    SHAPE = "shape"
    ICON = "icon"
    PATTERN = "pattern"
    CHART = "chart"
    DIAGRAM = "diagram"


@dataclass
class EvalCase(BaseEvalCase):
    """SVGGenerator-specific evaluation case."""
    task_type: SVGTask = SVGTask.SHAPE
    task_config: Any = field(default_factory=dict)


def create_shape_task() -> EvalCase:
    """Generate a simple SVG shape."""
    return EvalCase(
        task_id="shape_circle",
        task_type=SVGTask.SHAPE,
        prompt="Create a simple SVG with a red circle in the center",
        validators=["validate_svg_structure"],
        description="Generate an SVG with a red circle",
    )


def create_icon_task() -> EvalCase:
    """Generate an SVG icon."""
    return EvalCase(
        task_id="icon_star",
        task_type=SVGTask.ICON,
        prompt="Create an SVG icon of a 5-pointed star",
        validators=["validate_svg_structure"],
        description="Generate an SVG star icon",
    )


def create_pattern_task() -> EvalCase:
    """Generate an SVG pattern."""
    return EvalCase(
        task_id="pattern_grid",
        task_type=SVGTask.PATTERN,
        prompt="Create an SVG with a grid pattern of squares",
        validators=["validate_svg_structure"],
        description="Generate an SVG grid pattern",
    )


def create_chart_task() -> EvalCase:
    """Generate an SVG chart."""
    return EvalCase(
        task_id="chart_bar",
        task_type=SVGTask.CHART,
        prompt="Create a simple SVG bar chart with 3 bars",
        validators=["validate_svg_structure"],
        description="Generate an SVG bar chart",
    )


def create_diagram_task() -> EvalCase:
    """Generate an SVG diagram."""
    return EvalCase(
        task_id="diagram_flow",
        task_type=SVGTask.DIAGRAM,
        prompt="Create an SVG diagram showing a simple workflow with 3 connected boxes",
        validators=["validate_svg_structure"],
        description="Generate an SVG workflow diagram",
    )


def generate_eval_cases() -> List[EvalCase]:
    """Generate all evaluation cases."""
    return [
        create_shape_task(),
        create_icon_task(),
        create_pattern_task(),
        create_chart_task(),
        create_diagram_task(),
    ]


def validate_svg_structure(svg_elements: Any) -> Tuple[bool, str]:
    """Validate that SVG elements are generated."""
    if not svg_elements:
        return False, "No SVG elements generated"

    if not isinstance(svg_elements, list):
        return False, f"SVG elements is not a list: {type(svg_elements)}"

    if len(svg_elements) == 0:
        return False, "SVG elements list is empty"

    # Check first element has content attribute
    first = svg_elements[0]
    if not hasattr(first, 'content'):
        return False, f"First element doesn't have content attribute: {first}"

    content = str(first.content) if hasattr(first, 'content') else str(first)

    # Check it's actually SVG content
    if '<svg' not in content.lower():
        return False, f"Content doesn't appear to be SVG: {content[:100]}"

    if '</svg>' not in content.lower():
        return False, f"Content is not a complete SVG: {content[:100]}"

    return True, "SVG structure is valid"


def validate_svg_elements(svg_elements: Any) -> Tuple[bool, str]:
    """Validate that SVG contains expected elements."""
    if not svg_elements or not isinstance(svg_elements, list) or len(svg_elements) == 0:
        return False, "No SVG elements"

    first = svg_elements[0]
    content = str(first.content) if hasattr(first, 'content') else str(first)

    # Check for common SVG element types
    has_elements = any(tag in content.lower() for tag in [
        '<rect', '<circle', '<path', '<line', '<polygon', '<text', '<g'
    ])

    if not has_elements:
        return False, "SVG doesn't contain recognized graphic elements"

    return True, "SVG contains graphic elements"


class SVGGeneratorRunner(EvalRunner):
    """Runner for SVGGenerator evaluation."""

    def get_name(self) -> str:
        return "SVGGenerator"

    def get_default_models(self) -> List[Tuple[str, str]]:
        """Return default Ollama models."""
        return [
            ("ollama", "gemma3:4b"),
            ("ollama", "llama3.2:3b"),
        ]

    def get_eval_cases(self) -> List[EvalCase]:
        """Return all evaluation cases."""
        return generate_eval_cases()

    async def evaluate_case(
        self,
        eval_case: EvalCase,
        model: Tuple[str, str],
        context: ProcessingContext,
    ) -> EvalResult:
        """Evaluate a single case with a given model."""
        provider_name, model_id = model
        start_time = time.time()
        result_accuracy = 0.0
        error_msg = None

        try:
            # Get provider enum
            provider = get_provider(provider_name)

            # Create the SVGGenerator node
            node = SVGGenerator(
                model=LanguageModel(provider=provider, id=model_id),
                prompt=eval_case.prompt,
                max_tokens=4096,
            )

            # Run the generator
            svg_elements = await node.process(context)

            # Run validators
            validator_scores = []
            for validator_name in eval_case.validators:
                validator_func = globals().get(f"validate_{validator_name.split('validate_')[1]}")
                if validator_func is None:
                    validator_func = globals().get(validator_name)
                if validator_func:
                    is_valid, _ = validator_func(svg_elements)
                    validator_scores.append(1.0 if is_valid else 0.0)
                else:
                    validator_scores.append(0.0)

            # Calculate accuracy
            result_accuracy = sum(validator_scores) / len(validator_scores) if validator_scores else 0.0

        except Exception as e:
            error_msg = str(e)
            result_accuracy = 0.0

        runtime = time.time() - start_time

        return EvalResult(
            task_id=eval_case.task_id,
            model=f"{provider_name}/{model_id}",
            correct=result_accuracy > 0.7,
            accuracy_score=result_accuracy,
            runtime_seconds=runtime,
            error_message=error_msg,
            details=None,
        )


async def main_async(
    models: List[Tuple[str, str]],
    output_file: str | None,
) -> None:
    """Run evaluation asynchronously."""
    runner = SVGGeneratorRunner()
    await run_evaluation(runner, models or None, output_file)


def main():
    """Command-line interface for running evaluations."""
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(
        description="Evaluate SVGGenerator node with multiple language models",
        epilog="""
Examples:
  python eval_svg_generator.py
  python eval_svg_generator.py --models ollama/gemma3:4b
  python eval_svg_generator.py --output results.json
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models in format provider/model_id. "
        "Defaults to local Ollama models",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save results as JSON",
    )

    args = parser.parse_args()

    # Parse models if provided
    models: List[Tuple[str, str]] | None = None
    if args.models:
        models = []
        for model_str in args.models.split(","):
            parts = model_str.strip().split("/")
            if len(parts) == 2:
                models.append((parts[0], parts[1]))
            else:
                print(f"⚠️  Invalid model format: {model_str}. Use provider/model_id")

        if not models:
            print("❌ No valid models specified")
            return

    # Run evaluation
    asyncio.run(main_async(models or [], args.output))


if __name__ == "__main__":
    main()
