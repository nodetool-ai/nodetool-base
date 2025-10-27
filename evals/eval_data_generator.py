"""
Evaluation script for the DataGenerator node.

Tests the DataGenerator node with multiple language models on an eval dataset.
Calculates accuracy for each run and prints average accuracy per model.

Usage:
    python eval_data_generator.py
    python eval_data_generator.py --models ollama/gemma3:1b
    python eval_data_generator.py --output results.json
"""

import re
import time
from typing import Any, List, Tuple, Dict
from dataclasses import dataclass, field
from enum import Enum

from nodetool.metadata.types import (
    ColumnDef,
    LanguageModel,
    RecordType,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk
from nodetool.nodes.nodetool.generators import DataGenerator

try:
    from .common import EvalCase as BaseEvalCase, EvalResult, EvalRunner, run_evaluation, get_provider
except ImportError:
    from common import EvalCase as BaseEvalCase, EvalResult, EvalRunner, run_evaluation, get_provider


class GenerationTask(Enum):
    """Types of data generation tasks."""
    PEOPLE = "people"
    PRODUCTS = "products"
    NUMBERS = "numbers"
    EMAILS = "emails"
    CITIES = "cities"


@dataclass
class EvalCase(BaseEvalCase):
    """DataGenerator-specific evaluation case."""
    task_type: GenerationTask = GenerationTask.PEOPLE
    input_text: str = ""
    columns: RecordType = field(default_factory=RecordType)
    expected_rows: int = 0
    task_config: Dict[str, Any] = field(default_factory=dict)


def create_people_task() -> EvalCase:
    """Generate people with name and age."""
    return EvalCase(
        task_id="people_5",
        task_type=GenerationTask.PEOPLE,
        prompt="Generate 5 realistic people with diverse names and ages between 18 and 75",
        input_text="",
        columns=RecordType(
            columns=[
                ColumnDef(name="name", data_type="string"),
                ColumnDef(name="age", data_type="int"),
            ]
        ),
        expected_rows=5,
        validators=["validate_people"],
        description="Generate 5 people with name and age",
    )


def create_products_task() -> EvalCase:
    """Generate products with name, price, and category."""
    return EvalCase(
        task_id="products_3",
        task_type=GenerationTask.PRODUCTS,
        prompt="Generate 3 unique products with realistic names, prices (0-1000), and categories",
        input_text="",
        columns=RecordType(
            columns=[
                ColumnDef(name="name", data_type="string"),
                ColumnDef(name="price", data_type="float"),
                ColumnDef(name="category", data_type="string"),
            ]
        ),
        expected_rows=3,
        validators=["validate_products"],
        description="Generate 3 products with name, price, and category",
    )


def create_numbers_task() -> EvalCase:
    """Generate random numbers in a specific range."""
    return EvalCase(
        task_id="numbers_10",
        task_type=GenerationTask.NUMBERS,
        prompt="Generate 10 random decimal numbers between 0 and 100",
        input_text="",
        columns=RecordType(
            columns=[
                ColumnDef(name="value", data_type="float"),
            ]
        ),
        expected_rows=10,
        validators=["validate_numbers"],
        description="Generate 10 random numbers between 0 and 100",
    )


def create_emails_task() -> EvalCase:
    """Generate valid email addresses."""
    return EvalCase(
        task_id="emails_4",
        task_type=GenerationTask.EMAILS,
        prompt="Generate 4 realistic and valid email addresses with different domains",
        input_text="",
        columns=RecordType(
            columns=[
                ColumnDef(name="email", data_type="string"),
            ]
        ),
        expected_rows=4,
        validators=["validate_emails"],
        description="Generate 4 valid email addresses",
    )


def create_cities_task() -> EvalCase:
    """Generate cities and countries."""
    return EvalCase(
        task_id="cities_6",
        task_type=GenerationTask.CITIES,
        prompt="Generate 6 major world cities with their countries",
        input_text="",
        columns=RecordType(
            columns=[
                ColumnDef(name="city", data_type="string"),
                ColumnDef(name="country", data_type="string"),
            ]
        ),
        expected_rows=6,
        validators=["validate_cities"],
        description="Generate 6 cities with their countries",
    )


def generate_eval_cases() -> List[EvalCase]:
    """Generate all evaluation cases."""
    return [
        create_people_task(),
        create_products_task(),
        create_numbers_task(),
        create_emails_task(),
        create_cities_task(),
    ]


def validate_structure(data: List[List[Any]], columns: RecordType) -> Tuple[bool, str]:
    """Validate the basic structure of generated data."""
    if not data:
        return False, "No data generated"

    expected_cols = len(columns.columns)
    for row in data:
        if len(row) != expected_cols:
            return False, f"Row has {len(row)} columns, expected {expected_cols}"

    return True, "Structure is valid"


def validate_people(data: List[List[Any]], columns: RecordType) -> Tuple[bool, str]:
    """Validate generated people data."""
    is_valid, msg = validate_structure(data, columns)
    if not is_valid:
        return False, msg

    col_names = [c.name for c in columns.columns]
    name_idx = col_names.index("name") if "name" in col_names else -1
    age_idx = col_names.index("age") if "age" in col_names else -1

    for row in data:
        # Check name is not empty
        if name_idx >= 0:
            name = row[name_idx]
            if not name or not isinstance(name, str) or len(name.strip()) == 0:
                return False, f"Invalid name: {name}"

        # Check age is in reasonable range
        if age_idx >= 0:
            age = row[age_idx]
            if not isinstance(age, int):
                return False, f"Age is not int: {age}"
            if age < 0 or age > 150:
                return False, f"Age out of range: {age}"

    return True, "People data is valid"


def validate_products(data: List[List[Any]], columns: RecordType) -> Tuple[bool, str]:
    """Validate generated products data."""
    is_valid, msg = validate_structure(data, columns)
    if not is_valid:
        return False, msg

    col_names = [c.name for c in columns.columns]
    name_idx = col_names.index("name") if "name" in col_names else -1
    price_idx = col_names.index("price") if "price" in col_names else -1
    category_idx = col_names.index("category") if "category" in col_names else -1

    for row in data:
        # Check name
        if name_idx >= 0:
            name = row[name_idx]
            if not name or not isinstance(name, str) or len(name.strip()) == 0:
                return False, f"Invalid product name: {name}"

        # Check price
        if price_idx >= 0:
            price = row[price_idx]
            if not isinstance(price, (int, float)):
                return False, f"Price is not numeric: {price}"
            if price < 0 or price > 10000:
                return False, f"Price out of range: {price}"

        # Check category
        if category_idx >= 0:
            category = row[category_idx]
            if not category or not isinstance(category, str) or len(category.strip()) == 0:
                return False, f"Invalid category: {category}"

    return True, "Products data is valid"


def validate_numbers(data: List[List[Any]], columns: RecordType) -> Tuple[bool, str]:
    """Validate generated numbers data."""
    is_valid, msg = validate_structure(data, columns)
    if not is_valid:
        return False, msg

    col_names = [c.name for c in columns.columns]
    value_idx = col_names.index("value") if "value" in col_names else 0

    for row in data:
        value = row[value_idx]
        if not isinstance(value, (int, float)):
            return False, f"Value is not numeric: {value}"
        if value < 0 or value > 100:
            return False, f"Value out of range [0, 100]: {value}"

    return True, "Numbers data is valid"


def validate_emails(data: List[List[Any]], columns: RecordType) -> Tuple[bool, str]:
    """Validate generated email addresses."""
    is_valid, msg = validate_structure(data, columns)
    if not is_valid:
        return False, msg

    col_names = [c.name for c in columns.columns]
    email_idx = col_names.index("email") if "email" in col_names else 0

    # Simple email regex pattern
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

    for row in data:
        email = row[email_idx]
        if not isinstance(email, str):
            return False, f"Email is not a string: {email}"
        if not re.match(email_pattern, email):
            return False, f"Invalid email format: {email}"

    return True, "Emails data is valid"


def validate_cities(data: List[List[Any]], columns: RecordType) -> Tuple[bool, str]:
    """Validate generated cities data."""
    is_valid, msg = validate_structure(data, columns)
    if not is_valid:
        return False, msg

    col_names = [c.name for c in columns.columns]
    city_idx = col_names.index("city") if "city" in col_names else 0
    country_idx = col_names.index("country") if "country" in col_names else 1

    for row in data:
        # Check city
        city = row[city_idx]
        if not city or not isinstance(city, str) or len(city.strip()) == 0:
            return False, f"Invalid city: {city}"

        # Check country
        if len(row) > country_idx:
            country = row[country_idx]
            if not country or not isinstance(country, str) or len(country.strip()) == 0:
                return False, f"Invalid country: {country}"

    return True, "Cities data is valid"


class DataGeneratorRunner(EvalRunner):
    """Runner for DataGenerator evaluation."""

    def get_name(self) -> str:
        return "DataGenerator"

    def get_default_models(self) -> List[Tuple[str, str]]:
        """Return default Ollama models."""
        return [
            ("ollama", "gemma3:1b"),
            ("ollama", "gemma3:270m"),
            ("ollama", "gemma3:4b"),
            ("ollama", "qwen3:0.6b"),
            ("ollama", "qwen3:1.7b"),
            ("ollama", "qwen3:4b"),
            ("ollama", "qwen3:8b"),
            ("ollama", "llama3.2:3b"),
            ("ollama", "llama3.1:8b"),
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
        row_count = None

        try:
            # Get provider enum
            provider = get_provider(provider_name)

            # Create the DataGenerator node
            node = DataGenerator(
                model=LanguageModel(provider=provider, id=model_id),
                prompt=eval_case.prompt,
                input_text=eval_case.input_text,
                columns=eval_case.columns,
                max_tokens=4096,
            )

            # Collect generated data
            generated_rows = []
            async for output in node.gen_process(context):
                if output["record"] is not None:
                    generated_rows.append(output["record"])

            row_count = len(generated_rows)

            # Convert records to list format for validators
            data = []
            for record in generated_rows:
                col_names = [c.name for c in eval_case.columns.columns]
                row = [record.get(col_name) for col_name in col_names]
                data.append(row)

            # Run validators
            validator_scores = []
            for validator_name in eval_case.validators:
                validator_func = globals().get(f"validate_{validator_name.split('validate_')[1]}")
                if validator_func is None:
                    validator_func = globals().get(validator_name)
                if validator_func:
                    is_valid, _ = validator_func(data, eval_case.columns)
                    validator_scores.append(1.0 if is_valid else 0.0)
                else:
                    validator_scores.append(0.0)

            # Calculate accuracy based on validators and row count
            row_accuracy = 1.0 if row_count == eval_case.expected_rows else 0.5
            validator_accuracy = sum(validator_scores) / len(validator_scores) if validator_scores else 0.0
            result_accuracy = (row_accuracy + validator_accuracy) / 2.0

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
            details={"row_count": row_count, "expected_rows": eval_case.expected_rows},
        )


async def main_async(
    models: List[Tuple[str, str]],
    output_file: str | None,
) -> None:
    """Run evaluation asynchronously."""
    runner = DataGeneratorRunner()
    await run_evaluation(runner, models or None, output_file)


def main():
    """Command-line interface for running evaluations."""
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(
        description="Evaluate DataGenerator node with multiple language models",
        epilog="""
Examples:
  python eval_data_generator.py
  python eval_data_generator.py --models ollama/gemma3:1b
  python eval_data_generator.py --models openai/gpt-4,anthropic/claude-3-5-haiku-20241022
  python eval_data_generator.py --output results.json
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
