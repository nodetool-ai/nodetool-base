"""
Common evaluation framework for NodeTool evaluations.

Provides shared utilities for evaluating different node types:
- EvalCase: Base class for test cases
- EvalResult: Result tracking
- run_evaluation: Orchestration function
- Result formatting and export
"""

import json
import time
import traceback
from typing import Any, List, Tuple, Dict, Optional, Callable, Awaitable, Type
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import Provider


# Map provider strings to Provider enum values
PROVIDER_MAP = {
    provider.value: provider for provider in Provider.__members__.values()
}


@dataclass
class EvalCase:
    """Base class for a single evaluation case."""
    task_id: str
    prompt: str
    expected_output: Any = None
    validators: List[str] = field(default_factory=list)
    description: str = ""
    task_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of a single evaluation."""
    task_id: str
    model: str
    correct: bool
    accuracy_score: float
    runtime_seconds: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class EvalRunner(ABC):
    """Abstract base class for evaluation runners."""

    @abstractmethod
    async def evaluate_case(
        self,
        eval_case: EvalCase,
        model: Tuple[str, str],
        context: ProcessingContext,
    ) -> EvalResult:
        """Evaluate a single case with a given model."""
        pass

    @abstractmethod
    def get_eval_cases(self) -> List[EvalCase]:
        """Return list of evaluation cases."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return name of this evaluator (e.g., 'DataGenerator')."""
        pass

    @abstractmethod
    def get_default_models(self) -> List[Tuple[str, str]]:
        """Return default models to evaluate."""
        pass


async def run_evaluation(
    runner: EvalRunner,
    models: Optional[List[Tuple[str, str]]] = None,
    output_file: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Run evaluation on all cases with given models.

    Args:
        runner: EvalRunner instance with evaluation logic
        models: List of (provider, model_id) tuples. If None, uses runner defaults.
        output_file: Optional path to save results as JSON.

    Returns:
        Dictionary mapping model to list of accuracy scores.
    """
    if models is None:
        models = runner.get_default_models()

    eval_cases = runner.get_eval_cases()
    context = ProcessingContext(user_id="eval", auth_token="eval")

    all_results: List[EvalResult] = []
    model_accuracies: Dict[str, List[float]] = {f"{p}/{m}": [] for p, m in models}

    print(f"\n🚀 Starting {runner.get_name()} Evaluation")
    print(f"📊 Evaluating {len(eval_cases)} tasks")
    print(f"🤖 Models: {len(models)}")
    print(f"   {', '.join(f'{p}/{m}' for p, m in models)}\n")

    # Run evaluation for each case and model
    for case_idx, case in enumerate(eval_cases, 1):
        print(f"[{case_idx}/{len(eval_cases)}] {case.description}")
        for model_idx, model in enumerate(models, 1):
            result = await runner.evaluate_case(case, model, context)
            all_results.append(result)
            model_accuracies[result.model].append(result.accuracy_score)

            status = "✓" if result.correct else "✗"
            print(
                f"  {status} {model[0]}/{model[1]:<20} "
                f"accuracy={result.accuracy_score:.2%} "
                f"time={result.runtime_seconds:.2f}s"
            )
            if result.error_message:
                print(f"    Error: {result.error_message}")

    # Calculate and print statistics
    print("\n" + "=" * 70)
    print("📈 SUMMARY")
    print("=" * 70)

    model_stats: Dict[str, Dict[str, Any]] = {}
    for model, accuracies in model_accuracies.items():
        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            correct_count = sum(1 for acc in accuracies if acc > 0.7)
            model_stats[model] = {
                "average_accuracy": avg_accuracy,
                "correct_count": correct_count,
                "total_tests": len(accuracies),
                "accuracy_scores": accuracies,
            }
            print(
                f"\n{model}:"
                f"\n  Average Accuracy: {avg_accuracy:.2%}"
                f"\n  Correct Tests: {correct_count}/{len(accuracies)}"
            )

    # Find best and worst models
    if model_stats:
        best_model = max(model_stats.items(), key=lambda x: x[1]["average_accuracy"])
        print(f"\n🏆 Best Model: {best_model[0]} ({best_model[1]['average_accuracy']:.2%})")

    # Save results if requested
    if output_file:
        output_data = {
            "summary": model_stats,
            "detailed_results": [asdict(r) for r in all_results],
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Results saved to {output_file}")

    print("=" * 70 + "\n")

    return model_accuracies


def get_provider(provider_name: str) -> Provider:
    """
    Convert provider string to Provider enum.

    Args:
        provider_name: String like 'openai' or 'ollama'

    Returns:
        Provider enum value

    Raises:
        ValueError: If provider not found
    """
    if provider_name in PROVIDER_MAP:
        return PROVIDER_MAP[provider_name]

    # Try with uppercase
    upper_name = provider_name.upper()
    for provider in Provider:
        if provider.value == provider_name or provider.name == upper_name:
            return provider

    raise ValueError(f"Unknown provider: {provider_name}")


def safe_evaluate(
    func: Callable[..., Awaitable[EvalResult]],
) -> Callable[..., Awaitable[EvalResult]]:
    """
    Decorator to wrap evaluation functions with error handling.

    Catches exceptions, prints traceback, and returns error result.
    """
    async def wrapper(*args, **kwargs) -> EvalResult:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            print(f"Error in evaluation: {e}")

            # Extract task_id and model from args if available
            task_id = kwargs.get("eval_case", args[1] if len(args) > 1 else None)
            task_id = getattr(task_id, "task_id", "unknown")

            model = kwargs.get("model", args[2] if len(args) > 2 else ("unknown", "unknown"))
            model_str = f"{model[0]}/{model[1]}" if isinstance(model, tuple) else str(model)

            return EvalResult(
                task_id=task_id,
                model=model_str,
                correct=False,
                accuracy_score=0.0,
                runtime_seconds=0.0,
                error_message=str(e),
            )

    return wrapper


def validate_structure(
    data: Any,
    expected_type: Type,
    field_name: str = "data",
) -> Tuple[bool, str]:
    """
    Validate that data matches expected type.

    Args:
        data: The data to validate
        expected_type: Expected type or tuple of types
        field_name: Name of field for error messages

    Returns:
        (is_valid, error_message)
    """
    if data is None:
        return False, f"{field_name} is None"

    if not isinstance(data, expected_type):
        return False, f"{field_name} is {type(data).__name__}, expected {expected_type.__name__}"

    return True, "Structure is valid"


def validate_non_empty(
    data: Any,
    field_name: str = "data",
) -> Tuple[bool, str]:
    """
    Validate that data is non-empty.

    Args:
        data: The data to validate
        field_name: Name of field for error messages

    Returns:
        (is_valid, error_message)
    """
    if not data:
        return False, f"{field_name} is empty"

    return True, f"{field_name} is non-empty"


def calculate_accuracy(
    metrics: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Calculate weighted accuracy from metrics.

    Args:
        metrics: Dict of metric_name -> score (0-1)
        weights: Optional dict of metric_name -> weight.
                If None, equal weight for all metrics.

    Returns:
        Accuracy score (0-1)
    """
    if not metrics:
        return 0.0

    if weights is None:
        # Equal weight
        return sum(metrics.values()) / len(metrics)

    # Weighted average
    total_weight = sum(weights.get(k, 1.0) for k in metrics.keys())
    if total_weight == 0:
        return 0.0

    weighted_sum = sum(
        score * weights.get(k, 1.0)
        for k, score in metrics.items()
    )
    return weighted_sum / total_weight
