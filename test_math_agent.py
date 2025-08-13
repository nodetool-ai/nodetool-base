import asyncio
import os
import uuid
import math
import re
import time
import sys
import io
import contextlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Awaitable

from nodetool.agents.simple_agent import SimpleAgent
from nodetool.chat.providers.anthropic_provider import AnthropicProvider
from nodetool.chat.providers.base import ChatProvider
from nodetool.chat.providers.gemini_provider import GeminiProvider
from nodetool.chat.providers.huggingface_provider import PROVIDER_T, HuggingFaceProvider
from rich.live import Live
from rich.table import Table
from rich.columns import Columns

from nodetool.agents.agent import Agent
from nodetool.agents.tools.node_tool import NodeTool
from nodetool.chat.providers.openai_provider import OpenAIProvider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.math import BinaryOp, UnaryOp


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


@dataclass
class ModelStats:
    finished: int = 0
    correct: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_runtime_seconds: float = 0.0


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


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


@dataclass
class LogEntry:
    model: str
    problem: str
    result: float | None
    correct: bool | None
    runtime_seconds: float


def make_log_table(log_entries: List["LogEntry"]) -> Table:
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


def make_view(stats: Dict[str, ModelStats], log_entries: List["LogEntry"]):
    stats_table = make_table(stats)
    logs_table = make_log_table(log_entries)
    return Columns([stats_table, logs_table], equal=True, expand=True)


async def run_problem_for_model(
    provider: ChatProvider, model: str, problem_text: str
) -> Tuple[float | None, Usage]:
    context = ProcessingContext()
    usage: Usage = Usage()
    agent: SimpleAgent | None = None
    try:
        agent = SimpleAgent(
            name="Math Agent",
            objective=build_objective(problem_text),
            provider=provider,
            model=model,
            tools=[NodeTool(BinaryOp), NodeTool(UnaryOp)],
            output_schema={"value": "number"},
        )

        async for _ in agent.execute(context):
            # Suppress chunk output to keep the Rich table clean
            pass

        usage = Usage(
            input_tokens=int(getattr(agent.subtask_context, "input_tokens_total", 0)),
            output_tokens=int(getattr(agent.subtask_context, "output_tokens_total", 0)),
        )
        return agent.get_results().get("value", None), usage
    except Exception:
        # Treat any exception during agent execution as a failed task
        try:
            if agent is not None and agent.subtask_context is not None:
                usage = Usage(
                    input_tokens=int(agent.subtask_context.input_tokens_total),
                    output_tokens=int(agent.subtask_context.output_tokens_total),
                )
            else:
                usage = Usage()
        except Exception:
            pass
        return None, usage


def build_provider(provider_key: str) -> ChatProvider:
    if provider_key == "openai":
        return OpenAIProvider()
    if provider_key == "gemini":
        return GeminiProvider()
    if provider_key == "anthropic":
        return AnthropicProvider()
    if provider_key.startswith("huggingface:"):
        _, inference_provider = provider_key.split(":", 1)
        return HuggingFaceProvider(inference_provider=inference_provider)  # type: ignore[arg-type]
    raise ValueError(f"Unknown provider key: {provider_key}")


def execute_task_in_process(
    provider_key: str, model: str, problem_text: str
) -> Tuple[float | None, int, int, float]:
    # Capture all prints from provider libraries/tools to avoid stdout bleed
    stdout_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
        stdout_buffer
    ):
        start_time = time.perf_counter()
        provider = build_provider(provider_key)
        result, usage = asyncio.run(
            run_problem_for_model(provider, model, problem_text)
        )
        elapsed_seconds = time.perf_counter() - start_time
    safe_result = None if result is None else float(result)
    input_tokens = int(getattr(usage, "input_tokens", 0))
    output_tokens = int(getattr(usage, "output_tokens", 0))
    return safe_result, input_tokens, output_tokens, float(elapsed_seconds)


async def worker(
    lock: asyncio.Lock,
    stats: Dict[str, ModelStats],
    log_lines: List["LogEntry"],
    live: Live | None,
    executor: ProcessPoolExecutor,
    provider_key: str,
    model: str,
    problem_text: str,
    expected: float,
):
    loop = asyncio.get_running_loop()
    result, input_toks, output_toks, elapsed_seconds = await loop.run_in_executor(
        executor, execute_task_in_process, provider_key, model, problem_text
    )
    is_correct = False
    if result is not None:
        is_correct = math.isclose(result, float(expected), rel_tol=1e-6, abs_tol=1e-6)
    async with lock:
        s = stats[model]
        s.input_tokens += input_toks
        s.output_tokens += output_toks
        s.finished += 1
        if is_correct:
            s.correct += 1
        s.total_runtime_seconds += elapsed_seconds
        short_problem = (
            problem_text if len(problem_text) <= 80 else problem_text[:77] + "..."
        )
        log_lines.append(
            LogEntry(
                model=model,
                problem=short_problem,
                result=(None if result is None else float(result)),
                correct=(
                    True if is_correct else (False if result is not None else None)
                ),
                runtime_seconds=float(elapsed_seconds),
            )
        )
        if live is not None:
            live.update(make_view(stats, log_lines))


async def main():
    problems = generate_math_problems()
    stats: Dict[str, ModelStats] = {m: ModelStats() for p, m in MODELS}
    lock = asyncio.Lock()
    log_lines: List[LogEntry] = []

    # Concurrency across all (model, problem) pairs using process pool for isolation
    concurrency = int(os.getenv("MATH_AGENT_CONCURRENCY", "8"))
    executor = ProcessPoolExecutor(max_workers=concurrency)

    live = Live(make_view(stats, log_lines), refresh_per_second=8)
    live.start()
    # live = None
    try:
        tasks: List[asyncio.Task[None]] = []
        for problem_text, expected in problems:
            for provider_key, model in MODELS:
                tasks.append(
                    asyncio.create_task(
                        worker(
                            lock=lock,
                            stats=stats,
                            log_lines=log_lines,
                            live=live,
                            executor=executor,
                            provider_key=provider_key,
                            model=model,
                            problem_text=problem_text,
                            expected=expected,
                        )
                    )
                )
        await asyncio.gather(*tasks)
    finally:
        if live is not None:
            live.stop()
        executor.shutdown(wait=True)


async def test_agent(provider: ChatProvider, model: str, problem_text: str):
    context = ProcessingContext()
    agent = SimpleAgent(
        name="Math Agent",
        objective=build_objective(problem_text),
        provider=provider,
        model=model,
        tools=[NodeTool(BinaryOp), NodeTool(UnaryOp)],
        output_schema={"value": "number"},
    )

    async for chunk in agent.execute(context):
        print(chunk)

    print(agent.get_results())


if __name__ == "__main__":
    asyncio.run(main())
    # asyncio.run(
    #     test_agent(
    #         HuggingFaceProvider(inference_provider="cerebras"),
    #         # "openai/gpt-oss-120b",
    #         "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    #         "Compute square_root(square(12) + square(5))",
    #     )
    # )
