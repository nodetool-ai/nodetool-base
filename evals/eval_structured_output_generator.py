"""
Evaluation script for the StructuredOutputGenerator node.

Tests the StructuredOutputGenerator node for generating structured JSON from prompts.

Usage:
    python eval_structured_output_generator.py
    python eval_structured_output_generator.py --models ollama/gemma3:4b
    python eval_structured_output_generator.py --output results.json
"""

import json
import time
from typing import List, Tuple, Any, Dict
from dataclasses import dataclass, field
from enum import Enum

from nodetool.metadata.types import LanguageModel, ColumnDef, RecordType, TypeMetadata
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.nodetool.generators import StructuredOutputGenerator

try:
    from .common import EvalCase as BaseEvalCase, EvalResult, EvalRunner, run_evaluation, get_provider
except ImportError:
    from common import EvalCase as BaseEvalCase, EvalResult, EvalRunner, run_evaluation, get_provider


class StructuredTask(Enum):
    """Types of structured output tasks."""
    PERSON = "person"
    PRODUCT = "product"
    EVENT = "event"
    LOCATION = "location"
    ORGANIZATION = "organization"


@dataclass
class EvalCase(BaseEvalCase):
    """StructuredOutputGenerator-specific evaluation case."""
    task_type: StructuredTask = StructuredTask.PERSON
    output_schema: Dict[str, str] = field(default_factory=dict)
    output_columns: RecordType = field(default_factory=RecordType)
    task_config: Any = field(default_factory=dict)


def create_person_task() -> EvalCase:
    """Generate a person object."""
    return EvalCase(
        task_id="person",
        task_type=StructuredTask.PERSON,
        prompt="Generate a realistic person with a name, age, and email",
        output_columns=RecordType(columns=[
            ColumnDef(name="name", data_type="string"),
            ColumnDef(name="age", data_type="int"),
            ColumnDef(name="email", data_type="string"),
        ]),
        output_schema={
            "name": "string",
            "age": "int",
            "email": "string",
        },
        validators=["validate_person"],
        description="Generate a person with name, age, and email",
    )


def create_product_task() -> EvalCase:
    """Generate a product object."""
    return EvalCase(
        task_id="product",
        task_type=StructuredTask.PRODUCT,
        prompt="Generate a product with a name, price, and category",
        output_columns=RecordType(columns=[
            ColumnDef(name="name", data_type="string"),
            ColumnDef(name="price", data_type="float"),
            ColumnDef(name="category", data_type="string"),
        ]),
        output_schema={
            "name": "string",
            "price": "float",
            "category": "string",
        },
        validators=["validate_product"],
        description="Generate a product with name, price, and category",
    )


def create_event_task() -> EvalCase:
    """Generate an event object."""
    return EvalCase(
        task_id="event",
        task_type=StructuredTask.EVENT,
        prompt="Generate an event with a title, date, and location",
        output_columns=RecordType(columns=[
            ColumnDef(name="title", data_type="string"),
            ColumnDef(name="date", data_type="string"),
            ColumnDef(name="location", data_type="string"),
        ]),
        output_schema={
            "title": "string",
            "date": "string",
            "location": "string",
        },
        validators=["validate_event"],
        description="Generate an event with title, date, and location",
    )


def create_location_task() -> EvalCase:
    """Generate a location object."""
    return EvalCase(
        task_id="location",
        task_type=StructuredTask.LOCATION,
        prompt="Generate a location with a city, country, and population",
        output_columns=RecordType(columns=[
            ColumnDef(name="city", data_type="string"),
            ColumnDef(name="country", data_type="string"),
            ColumnDef(name="population", data_type="int"),
        ]),
        output_schema={
            "city": "string",
            "country": "string",
            "population": "int",
        },
        validators=["validate_location"],
        description="Generate a location with city, country, and population",
    )


def create_organization_task() -> EvalCase:
    """Generate an organization object."""
    return EvalCase(
        task_id="organization",
        task_type=StructuredTask.ORGANIZATION,
        prompt="Generate an organization with a name, industry, and employee count",
        output_columns=RecordType(columns=[
            ColumnDef(name="name", data_type="string"),
            ColumnDef(name="industry", data_type="string"),
            ColumnDef(name="employees", data_type="int"),
        ]),
        output_schema={
            "name": "string",
            "industry": "string",
            "employees": "int",
        },
        validators=["validate_organization"],
        description="Generate an organization with name, industry, and employees",
    )


def generate_eval_cases() -> List[EvalCase]:
    """Generate all evaluation cases."""
    return [
        create_person_task(),
        create_product_task(),
        create_event_task(),
        create_location_task(),
        create_organization_task(),
    ]


def validate_is_dict(obj: Any) -> Tuple[bool, str]:
    """Validate that output is a dictionary."""
    if not isinstance(obj, dict):
        return False, f"Output is not a dict: {type(obj)}"

    if not obj:
        return False, "Output dictionary is empty"

    return True, "Output is a valid dictionary"


def validate_has_keys(obj: Dict, expected_keys: List[str]) -> Tuple[bool, str]:
    """Validate that dictionary has expected keys."""
    missing_keys = [k for k in expected_keys if k not in obj]
    if missing_keys:
        return False, f"Missing keys: {missing_keys}"

    return True, f"Has all expected keys: {expected_keys}"


def validate_non_empty_strings(obj: Dict, string_keys: List[str]) -> Tuple[bool, str]:
    """Validate that specified keys have non-empty string values."""
    for key in string_keys:
        if key not in obj:
            return False, f"Missing key: {key}"

        value = obj[key]
        if not isinstance(value, str) or not value.strip():
            return False, f"Key {key} is not a non-empty string: {value}"

    return True, f"String fields {string_keys} are valid"


def validate_person(obj: Any) -> Tuple[bool, str]:
    """Validate generated person object."""
    is_valid, msg = validate_is_dict(obj)
    if not is_valid:
        return False, msg

    is_valid, msg = validate_has_keys(obj, ["name", "age", "email"])
    if not is_valid:
        return False, msg

    # Check name and email are non-empty strings
    if not isinstance(obj.get("name"), str) or not obj.get("name", "").strip():
        return False, "Name must be a non-empty string"

    if not isinstance(obj.get("email"), str) or not obj.get("email", "").strip():
        return False, "Email must be a non-empty string"

    # Check age is a reasonable int
    try:
        age = int(obj["age"])
        if not (0 <= age <= 150):
            return False, f"Age out of range: {age}"
    except (ValueError, TypeError):
        return False, f"Age is not a valid integer: {obj['age']}"

    return True, "Person object is valid"


def validate_product(obj: Any) -> Tuple[bool, str]:
    """Validate generated product object."""
    is_valid, msg = validate_is_dict(obj)
    if not is_valid:
        return False, msg

    is_valid, msg = validate_has_keys(obj, ["name", "price", "category"])
    if not is_valid:
        return False, msg

    # Check name and category are non-empty strings
    if not isinstance(obj.get("name"), str) or not obj.get("name", "").strip():
        return False, "Name must be a non-empty string"

    if not isinstance(obj.get("category"), str) or not obj.get("category", "").strip():
        return False, "Category must be a non-empty string"

    # Check price is a reasonable float
    try:
        price = float(obj["price"])
        if not (0 <= price <= 100000):
            return False, f"Price out of range: {price}"
    except (ValueError, TypeError):
        return False, f"Price is not a valid number: {obj['price']}"

    return True, "Product object is valid"


def validate_event(obj: Any) -> Tuple[bool, str]:
    """Validate generated event object."""
    is_valid, msg = validate_is_dict(obj)
    if not is_valid:
        return False, msg

    is_valid, msg = validate_has_keys(obj, ["title", "date", "location"])
    if not is_valid:
        return False, msg

    # Check all fields are non-empty strings
    for field in ["title", "date", "location"]:
        if not isinstance(obj.get(field), str) or not obj.get(field, "").strip():
            return False, f"{field} must be a non-empty string"

    return True, "Event object is valid"


def validate_location(obj: Any) -> Tuple[bool, str]:
    """Validate generated location object."""
    is_valid, msg = validate_is_dict(obj)
    if not is_valid:
        return False, msg

    is_valid, msg = validate_has_keys(obj, ["city", "country", "population"])
    if not is_valid:
        return False, msg

    # Check city and country are non-empty strings
    if not isinstance(obj.get("city"), str) or not obj.get("city", "").strip():
        return False, "City must be a non-empty string"

    if not isinstance(obj.get("country"), str) or not obj.get("country", "").strip():
        return False, "Country must be a non-empty string"

    # Check population is a reasonable int
    try:
        pop = int(obj["population"])
        if not (0 <= pop <= 2000000000):  # Max world population
            return False, f"Population out of range: {pop}"
    except (ValueError, TypeError):
        return False, f"Population is not a valid integer: {obj['population']}"

    return True, "Location object is valid"


def validate_organization(obj: Any) -> Tuple[bool, str]:
    """Validate generated organization object."""
    is_valid, msg = validate_is_dict(obj)
    if not is_valid:
        return False, msg

    is_valid, msg = validate_has_keys(obj, ["name", "industry", "employees"])
    if not is_valid:
        return False, msg

    # Check name and industry are non-empty strings
    if not isinstance(obj.get("name"), str) or not obj.get("name", "").strip():
        return False, "Name must be a non-empty string"

    if not isinstance(obj.get("industry"), str) or not obj.get("industry", "").strip():
        return False, "Industry must be a non-empty string"

    # Check employees is a reasonable int
    try:
        emp = int(obj["employees"])
        if not (0 <= emp <= 10000000):  # Max reasonable company size
            return False, f"Employees out of range: {emp}"
    except (ValueError, TypeError):
        return False, f"Employees is not a valid integer: {obj['employees']}"

    return True, "Organization object is valid"

def type_name(data_type: str) -> str:
    if data_type == "int":
        return "int"
    elif data_type == "float":
        return "float"
    elif data_type == "string":
        return "str"
    elif data_type == "datetime":
        return "datetime"
    elif data_type == "object":
        return "dict"
    else:
        raise ValueError(f"Unknown data type: {data_type}")


class StructuredOutputGeneratorRunner(EvalRunner):
    """Runner for StructuredOutputGenerator evaluation."""

    def get_name(self) -> str:
        return "StructuredOutputGenerator"

    def get_default_models(self) -> List[Tuple[str, str]]:
        """Return default Ollama models."""
        return [
            ("ollama", "gemma3:1b"),
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

        try:
            # Get provider enum
            provider = get_provider(provider_name)

            # Build dynamic output slots from the schema
            # For StructuredOutputGenerator, we need to configure output slots
            node = StructuredOutputGenerator(
                model=LanguageModel(provider=provider, id=model_id),
                context=eval_case.prompt,
                instructions=f"Generate a JSON object matching this structure: {json.dumps(eval_case.output_schema)}",
                system_prompt="You are a JSON generator. Output ONLY valid JSON.",
                max_tokens=2048,
            )

            # Manually add output slots (this is normally done in the UI)
            for col in eval_case.output_columns.columns:
                
                node.dynamic_outputs[col.name] = TypeMetadata(type=type_name(col.data_type))

            # Run the generator - it's not async, so we need to handle it differently
            # For now, we'll try to call process()
            result = await node.process(context)

            if not isinstance(result, dict):
                return EvalResult(
                    task_id=eval_case.task_id,
                    model=f"{provider_name}/{model_id}",
                    correct=False,
                    accuracy_score=0.0,
                    runtime_seconds=time.time() - start_time,
                    error_message="Result is not a dictionary",
                    details=None,
                )

            # Run validators
            validator_scores = []
            for validator_name in eval_case.validators:
                validator_func = globals().get(f"validate_{validator_name.split('validate_')[1]}")
                if validator_func is None:
                    validator_func = globals().get(validator_name)
                if validator_func:
                    is_valid, _ = validator_func(result)
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
    runner = StructuredOutputGeneratorRunner()
    await run_evaluation(runner, models or None, output_file)


def main():
    """Command-line interface for running evaluations."""
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(
        description="Evaluate StructuredOutputGenerator node with multiple language models",
        epilog="""
Examples:
  python eval_structured_output_generator.py
  python eval_structured_output_generator.py --models ollama/gemma3:4b
  python eval_structured_output_generator.py --output results.json
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
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
