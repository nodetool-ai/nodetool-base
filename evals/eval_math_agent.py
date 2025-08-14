import asyncio
import os
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from rich.table import Table
from rich.columns import Columns
from rich.console import Console
from rich.live import Live

from nodetool.agents.simple_agent import SimpleAgent
from nodetool.agents.tools.node_tool import NodeTool
from nodetool.nodes.lib.math import BinaryOp, UnaryOp
from nodetool.agents.agent_evaluator import AgentEvaluator, ModelStats, EvaluationResult


MODELS: List[Tuple[str, str]] = [
    ("openai", "gpt-5"),
    ("openai", "gpt-5-mini"),
    ("gemini", "gemini-2.5-flash"),
    ("gemini", "gemini-2.5-flash-lite"),
    ("anthropic", "claude-sonnet-4-20250514"),
    ("anthropic", "claude-3-5-haiku-20241022"),
    ("huggingface:cerebras", "openai/gpt-oss-120b"),
    ("huggingface:cerebras", "Qwen/Qwen3-Coder-480B-A35B-Instruct"),
]


# Keep a modest rolling buffer of recent agent results in the log panel
MAX_LOG_LINES: int = int(os.getenv("MATH_AGENT_LOG_LINES", "50"))


def generate_math_problems() -> List[Tuple[str, float]]:
    """Return a suite of deterministic, multi-step problems solvable via UnaryOp and BinaryOp only."""
    problems: List[Tuple[str, float]] = []

    # Multi-step arithmetic, roots, powers (square/cube), trig, inverse trig, log, and modulus
    problems.append(
        (
            "Compute square_root(square(12) + square(5))",
            math.sqrt(12 * 12 + 5 * 5),
        )
    )  # 13

    problems.append(
        (
            "Compute square(square_root(2) + square_root(8))",
            (math.sqrt(2) + math.sqrt(8)) ** 2,
        )
    )  # 18

    problems.append(
        (
            "Compute cube_root(27) + cube(2) - square(5)",
            (math.copysign(abs(27) ** (1 / 3), 27)) + (2 * 2 * 2) - (5 * 5),
        )
    )  # -14

    problems.append(
        (
            "Compute square(sine(0.7)) + square(cosine(0.7))",
            (math.sin(0.7) ** 2) + (math.cos(0.7) ** 2),
        )
    )  # ~1

    problems.append(
        (
            "Compute arctan(1) + arctan(1)",
            math.atan(1) + math.atan(1),
        )
    )  # ~pi/2

    problems.append(
        (
            "Compute log(64) / log(2)",
            math.log(64) / math.log(2),
        )
    )  # 6

    problems.append(
        (
            "Compute square_root((3 + 5) * (7 + 9))",
            math.sqrt((3 + 5) * (7 + 9)),
        )
    )  # sqrt(128)

    problems.append(
        (
            "Compute absolute(negate(42)) + (1001 modulus 97)",
            abs(-42) + (1001 % 97),
        )
    )  # 42 + 31 = 73

    problems.append(
        (
            "Compute (square(15) - square(9)) / square_root(36)",
            (15 * 15 - 9 * 9) / math.sqrt(36),
        )
    )  # 24

    problems.append(
        (
            "Compute cosine(0) + sine(0) + tangent(0)",
            math.cos(0.0) + math.sin(0.0) + math.tan(0.0),
        )
    )  # 1

    problems.append(
        (
            "Compute cube_root(125) + cube_root(64) + cube_root(1)",
            (math.copysign(abs(125) ** (1 / 3), 125))
            + (math.copysign(abs(64) ** (1 / 3), 64))
            + (math.copysign(abs(1) ** (1 / 3), 1)),
        )
    )  # 5 + 4 + 1 = 10

    problems.append(
        (
            "Compute square(square_root(50) - square_root(2))",
            (math.sqrt(50) - math.sqrt(2)) ** 2,
        )
    )  # 32

    problems.append(
        (
            "Compute square_root(square(7.5) + square(2.5))",
            math.sqrt((7.5 * 7.5) + (2.5 * 2.5)),
        )
    )  # sqrt(62.5)

    problems.append(
        (
            "Compute arccos(0.5) + arcsin(0.5)",
            math.acos(0.5) + math.asin(0.5),
        )
    )  # ~pi/2

    problems.append(
        (
            "Compute square_root(square(3) + square(4) + square(12))",
            math.sqrt(3 * 3 + 4 * 4 + 12 * 12),
        )
    )  # 13

    problems.append(
        (
            "Compute log(1) + square_root(81) / 3",
            math.log(1) + math.sqrt(81) / 3,
        )
    )  # 0 + 9/3 = 3

    problems.append(
        (
            "Compute square_root(2) * square_root(18)",
            math.sqrt(2) * math.sqrt(18),
        )
    )  # 6

    problems.append(
        (
            "Compute absolute(-3.75) + square_root(square(1.2))",
            abs(-3.75) + math.sqrt(1.2 * 1.2),
        )
    )  # 3.75 + 1.2 = 4.95

    problems.append(
        (
            "Compute (1000 modulus 37) * (50 modulus 7)",
            (1000 % 37) * (50 % 7),
        )
    )  # 1 * 1 = 1

    problems.append(
        (
            "Compute square(cube_root(8) + square_root(9))",
            ((math.copysign(abs(8) ** (1 / 3), 8)) + math.sqrt(9)) ** 2,
        )
    )  # (2 + 3)^2 = 25

    problems.append(
        (
            "Compute square_root(square_root(81) + cube_root(27))",
            math.sqrt(math.sqrt(81) + (math.copysign(abs(27) ** (1 / 3), 27))),
        )
    )  # sqrt(9 + 3) = sqrt(12)

    return problems


def build_objective(problem_text: str) -> str:
    """Constrain the agent to use provided tools and return only the numeric result."""
    return f"""
        Use ONLY the provided tools 'BinaryOp' and 'UnaryOp' via tool calls to compute the value.
        Provide the final result as a number in the 'value' field.
        Problem: {problem_text}
    """


def make_table(stats: Dict[str, ModelStats]) -> Table:
    table = Table()
    table.add_column("Model")
    table.add_column("Finished Tests", justify="right")
    table.add_column("Correct Results", justify="right")
    table.add_column("Input Tokens", justify="right")
    table.add_column("Output Tokens", justify="right")
    table.add_column("Avg Runtime (s)", justify="right")
    for provider_key, model in MODELS:
        s = stats[model]
        avg_runtime = (s.total_runtime_seconds / s.finished) if s.finished > 0 else 0.0
        table.add_row(
            model,
            str(s.finished),
            str(s.correct),
            str(s.input_tokens),
            str(s.output_tokens),
            f"{avg_runtime:.2f}",
        )
    return table


def make_log_table(log_entries: List[Any]) -> Table:
    table = Table(title="Agent Results")
    table.add_column("Model")
    table.add_column("Problem")
    table.add_column("Result", justify="right")
    table.add_column("Correct", justify="center")
    table.add_column("Runtime (s)", justify="right")
    if not log_entries:
        return table
    for entry in log_entries[-MAX_LOG_LINES:]:
        status = (
            "✓" if entry.correct is True else ("✗" if entry.correct is False else "—")
        )
        result_text = "None" if entry.result is None else str(entry.result)
        table.add_row(
            entry.model,
            entry.problem,
            result_text,
            status,
            f"{entry.runtime_seconds:.2f}",
        )
    return table


def make_view(stats: Dict[str, ModelStats], log_entries: List[Any]):
    stats_table = make_table(stats)
    logs_table = make_log_table(log_entries)
    return Columns([stats_table, logs_table], equal=True, expand=True)


def build_math_agent(
    provider: Any, model: str, tools: Sequence[Any], problem_text: str
) -> Any:
    return SimpleAgent(
        name="Math Agent",
        objective=build_objective(problem_text),
        provider=provider,
        model=model,
        tools=list(tools),
        output_schema={"value": "number"},
    )


def numeric_result_checker(result: Any, expected: Any) -> bool:
    try:
        if expected is None:
            return result is not None
        candidate: Any
        if isinstance(result, dict) and "value" in result:
            candidate = result["value"]
        else:
            candidate = result
        if candidate is None:
            return False
        if isinstance(candidate, (int, float)):
            value_f = float(candidate)
        elif isinstance(candidate, str):
            value_f = float(candidate)
        else:
            return False
        return math.isclose(value_f, float(expected), rel_tol=1e-6, abs_tol=1e-6)
    except Exception:
        return False


async def main():
    problems = generate_math_problems()
    tools = [NodeTool(BinaryOp), NodeTool(UnaryOp)]
    concurrency = int(os.getenv("MATH_AGENT_CONCURRENCY", "8"))

    evaluator = AgentEvaluator(
        models=MODELS,
        problems=problems,
        build_agent_fn=build_math_agent,
        result_checker=numeric_result_checker,
        tools=tools,
        concurrency=concurrency,
    )

    stats: Dict[str, ModelStats] = {m: ModelStats() for _, m in MODELS}
    log_lines: List[Any] = []

    with Live(make_view(stats, log_lines), refresh_per_second=8) as live:
        evaluator.on_update = lambda s, l: live.update(make_view(s, l))  # type: ignore
        result: EvaluationResult = await evaluator.evaluate()
        live.update(make_view(result.stats, result.logs))


if __name__ == "__main__":
    asyncio.run(main())
