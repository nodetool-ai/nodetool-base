"""
Evaluation script for the ListGenerator node.

Tests the ListGenerator node for generating lists of items from prompts.

Usage:
    python eval_list_generator.py
    python eval_list_generator.py --models ollama/gemma3:1b
    python eval_list_generator.py --output results.json
"""

import re
import time
from typing import List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from nodetool.metadata.types import LanguageModel
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk
from nodetool.nodes.nodetool.generators import ListGenerator

try:
    from .common import EvalCase as BaseEvalCase, EvalResult, EvalRunner, run_evaluation, get_provider
except ImportError:
    from common import EvalCase as BaseEvalCase, EvalResult, EvalRunner, run_evaluation, get_provider


class ListTask(Enum):
    """Types of list generation tasks."""
    MOVIES = "movies"
    FOODS = "foods"
    SKILLS = "skills"
    COUNTRIES = "countries"
    BOOKS = "books"


@dataclass
class EvalCase(BaseEvalCase):
    """ListGenerator-specific evaluation case."""
    task_type: ListTask = ListTask.MOVIES
    expected_count: int = 0
    task_config: Any = field(default_factory=dict)


def create_movies_task() -> EvalCase:
    """Generate movie titles."""
    return EvalCase(
        task_id="movies_5",
        task_type=ListTask.MOVIES,
        prompt="Generate 5 famous movie titles",
        expected_count=5,
        validators=["validate_movies"],
        description="Generate 5 famous movie titles",
    )


def create_foods_task() -> EvalCase:
    """Generate food names."""
    return EvalCase(
        task_id="foods_4",
        task_type=ListTask.FOODS,
        prompt="Generate 4 popular food dishes from around the world",
        expected_count=4,
        validators=["validate_foods"],
        description="Generate 4 popular food dishes",
    )


def create_skills_task() -> EvalCase:
    """Generate skill names."""
    return EvalCase(
        task_id="skills_6",
        task_type=ListTask.SKILLS,
        prompt="Generate 6 important professional skills for software engineers",
        expected_count=6,
        validators=["validate_skills"],
        description="Generate 6 professional skills",
    )


def create_countries_task() -> EvalCase:
    """Generate country names."""
    return EvalCase(
        task_id="countries_5",
        task_type=ListTask.COUNTRIES,
        prompt="Generate 5 countries known for beautiful landscapes",
        expected_count=5,
        validators=["validate_countries"],
        description="Generate 5 countries with beautiful landscapes",
    )


def create_books_task() -> EvalCase:
    """Generate book titles."""
    return EvalCase(
        task_id="books_3",
        task_type=ListTask.BOOKS,
        prompt="Generate 3 classic novels",
        expected_count=3,
        validators=["validate_books"],
        description="Generate 3 classic novels",
    )


def generate_eval_cases() -> List[EvalCase]:
    """Generate all evaluation cases."""
    return [
        create_movies_task(),
        create_foods_task(),
        create_skills_task(),
        create_countries_task(),
        create_books_task(),
    ]


def validate_structure(items: List[str]) -> Tuple[bool, str]:
    """Validate basic structure of generated list."""
    if not isinstance(items, list):
        return False, f"Items is not a list: {type(items)}"

    if not items:
        return False, "List is empty"

    for item in items:
        if not isinstance(item, str) or not item.strip():
            return False, f"Invalid item (not non-empty string): {item}"

    return True, "Structure is valid"


def validate_movies(items: List[str]) -> Tuple[bool, str]:
    """Validate generated movie titles."""
    is_valid, msg = validate_structure(items)
    if not is_valid:
        return False, msg

    # Check that items are reasonable movie titles (not empty, not too short)
    for item in items:
        if len(item.strip()) < 2:
            return False, f"Movie title too short: '{item}'"

    return True, "Movie titles are valid"


def validate_foods(items: List[str]) -> Tuple[bool, str]:
    """Validate generated food names."""
    is_valid, msg = validate_structure(items)
    if not is_valid:
        return False, msg

    # Food names should be reasonable
    for item in items:
        if len(item.strip()) < 2:
            return False, f"Food name too short: '{item}'"

    return True, "Food names are valid"


def validate_skills(items: List[str]) -> Tuple[bool, str]:
    """Validate generated skill names."""
    is_valid, msg = validate_structure(items)
    if not is_valid:
        return False, msg

    # Skills should be reasonable length
    for item in items:
        if len(item.strip()) < 2:
            return False, f"Skill name too short: '{item}'"

    return True, "Skill names are valid"


def validate_countries(items: List[str]) -> Tuple[bool, str]:
    """Validate generated country names."""
    is_valid, msg = validate_structure(items)
    if not is_valid:
        return False, msg

    # Countries should have reasonable names
    for item in items:
        if len(item.strip()) < 2:
            return False, f"Country name too short: '{item}'"

    return True, "Country names are valid"


def validate_books(items: List[str]) -> Tuple[bool, str]:
    """Validate generated book titles."""
    is_valid, msg = validate_structure(items)
    if not is_valid:
        return False, msg

    # Book titles should be reasonable
    for item in items:
        if len(item.strip()) < 2:
            return False, f"Book title too short: '{item}'"

    return True, "Book titles are valid"


class ListGeneratorRunner(EvalRunner):
    """Runner for ListGenerator evaluation."""

    def get_name(self) -> str:
        return "ListGenerator"

    def get_default_models(self) -> List[Tuple[str, str]]:
        """Return default Ollama models."""
        return [
            ("ollama", "gemma3:1b"),
            ("ollama", "gemma3:270m"),
            ("ollama", "gemma3:4b"),
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
        item_count = 0

        try:
            # Get provider enum
            provider = get_provider(provider_name)

            # Create the ListGenerator node
            node = ListGenerator(
                model=LanguageModel(provider=provider, id=model_id),
                prompt=eval_case.prompt,
                input_text="",
                max_tokens=2048,
            )

            # Collect generated items
            collected_items: list[str] = []
            async for output in node.gen_process(context):
                if isinstance(output, dict) and "item" in output:
                    collected_items.append(output["item"])

            item_count = len(collected_items)

            # Run validators
            validator_scores = []
            for validator_name in eval_case.validators:
                validator_func = globals().get(f"validate_{validator_name.split('validate_')[1]}")
                if validator_func is None:
                    validator_func = globals().get(validator_name)
                if validator_func:
                    is_valid, _ = validator_func(collected_items)
                    validator_scores.append(1.0 if is_valid else 0.0)
                else:
                    validator_scores.append(0.0)

            # Calculate accuracy based on item count and validators
            count_accuracy = 1.0 if item_count == eval_case.expected_count else 0.8 if item_count > 0 else 0.0
            validator_accuracy = sum(validator_scores) / len(validator_scores) if validator_scores else 0.0
            result_accuracy = (count_accuracy + validator_accuracy) / 2.0

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
            details={"item_count": item_count, "expected_count": eval_case.expected_count},
        )


async def main_async(
    models: List[Tuple[str, str]],
    output_file: str | None,
) -> None:
    """Run evaluation asynchronously."""
    runner = ListGeneratorRunner()
    await run_evaluation(runner, models or None, output_file)


def main():
    """Command-line interface for running evaluations."""
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(
        description="Evaluate ListGenerator node with multiple language models",
        epilog="""
Examples:
  python eval_list_generator.py
  python eval_list_generator.py --models ollama/gemma3:1b
  python eval_list_generator.py --output results.json
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
