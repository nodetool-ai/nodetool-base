import argparse
import json
import os
import sys
import asyncio
import threading
from typing import Any, Awaitable, Callable, Dict, List, Tuple, TypeVar
from nodetool.providers import get_provider
from nodetool.providers.base import BaseProvider
from rich.table import Table
from rich.columns import Columns
from nodetool.agents.agent_evaluator import (
    AgentEvaluator,
    ModelStats,
    EvaluationResult,
)
from nodetool.workflows.processing_context import ProcessingContext


# Ensure local eval modules are importable as plain module names
THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import eval_data_agent as data_eval  # type: ignore
import eval_math_agent as math_eval  # type: ignore
import eval_browser_agent as browser_eval  # type: ignore
import eval_search_agent as search_eval  # type: ignore


MODELS: List[Tuple[str, str]] = [
    # ("openai", "gpt-5"),
    # ("openai", "gpt-5-mini"),
    # ("gemini", "gemini-2.5-flash"),
    # ("gemini", "gemini-2.5-flash-lite"),
    # ("anthropic", "claude-sonnet-4-20250514"),
    # ("anthropic", "claude-3-5-haiku-20241022"),
    # ("huggingface_cerebras", "openai/gpt-oss-120b"),
    ("ollama", "qwen3:4b"),
]

T = TypeVar("T")


def make_table(stats: Dict[str, ModelStats], models: List[Tuple[str, str]]) -> Table:
    """Create a consolidated stats table for any agent type."""
    table = Table()
    table.add_column("Model")
    table.add_column("Finished Tests", justify="right")
    table.add_column("Correct Results", justify="right")
    table.add_column("Input Tokens", justify="right")
    table.add_column("Output Tokens", justify="right")
    table.add_column("Avg Runtime (s)", justify="right")
    for _, model in models:
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


def make_log_table(
    log_entries: List[Any],
    max_lines: int = 50,
    problem_column_name: str = "Problem",
    truncate_long_text: bool = False,
) -> Table:
    """Create a consolidated log table for any agent type."""
    table = Table(title="Agent Results")
    table.add_column("Model")
    table.add_column(problem_column_name)
    table.add_column("Result", justify="right")
    table.add_column("Correct", justify="center")
    table.add_column("Runtime (s)", justify="right")
    if not log_entries:
        return table
    for entry in log_entries[-max_lines:]:
        status = (
            "✓" if entry.correct is True else ("✗" if entry.correct is False else "—")
        )

        # Handle result text formatting
        result_text = "None" if entry.result is None else str(entry.result)
        if truncate_long_text and len(result_text) > 50:
            result_text = result_text[:50] + "..."

        # Handle problem text formatting
        problem_text = entry.problem
        if truncate_long_text and len(problem_text) > 30:
            problem_text = problem_text[:30] + "..."

        table.add_row(
            entry.model,
            problem_text,
            result_text,
            status,
            f"{entry.runtime_seconds:.2f}",
        )
    return table


def make_view(
    stats: Dict[str, ModelStats],
    log_entries: List[Any],
    models: List[Tuple[str, str]],
    max_log_lines: int = 50,
    problem_column_name: str = "Problem",
    truncate_long_text: bool = False,
) -> Columns:
    """Create a consolidated view for any agent type."""
    stats_table = make_table(stats, models)
    logs_table = make_log_table(
        log_entries, max_log_lines, problem_column_name, truncate_long_text
    )
    return Columns([stats_table, logs_table], equal=True, expand=True)


async def _execute_agent_once(agent: Any, output_json_path: str | None) -> None:
    context = ProcessingContext()
    async for _ in agent.execute(context):
        pass
    result = agent.get_results()
    input_tokens = int(getattr(agent.subtask_context, "input_tokens_total", 0))
    output_tokens = int(getattr(agent.subtask_context, "output_tokens_total", 0))
    payload = {
        "result": result,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)


def _build_math_agent(provider: BaseProvider, model: str, problem_payload: Any) -> Any:
    from nodetool.agents.tools.node_tool import NodeTool
    from nodetool.nodes.lib.math import (
        Add,
        Subtract,
        Multiply,
        Divide,
        Modulus,
        MathFunction,
    )

    tools = [
        NodeTool(Add),
        NodeTool(Subtract),
        NodeTool(Multiply),
        NodeTool(Divide),
        NodeTool(Modulus),
        NodeTool(MathFunction),
    ]
    return math_eval.build_math_agent(provider, model, tools, problem_payload)


def _build_data_agent(provider: BaseProvider, model: str, problem_payload: Any) -> Any:
    # Reuse exported tools and builder from data eval
    return data_eval.build_data_agent(provider, model, data_eval.tools, problem_payload)


def _build_browser_agent(provider: BaseProvider, model: str, problem_payload: Any) -> Any:
    from nodetool.agents.tools.browser_tools import BrowserTool

    # Expect problem_payload as (description, url)
    if isinstance(problem_payload, (list, tuple)) and len(problem_payload) >= 2:
        task_description, url = problem_payload[0], problem_payload[1]
    elif isinstance(problem_payload, str) and "|" in problem_payload:
        task_description, url = problem_payload.split("|", 1)
    else:
        raise ValueError(
            "Invalid problem payload for browser agent; expected [desc, url]"
        )
    tools = [BrowserTool()]
    return browser_eval.build_browser_agent(
        provider, model, tools, (task_description, url)
    )


def _build_search_agent(provider: BaseProvider, model: str, problem_payload: Any) -> Any:
    from nodetool.agents.tools.node_tool import NodeTool
    from nodetool.nodes.search.google import (
        GoogleSearch,
        GoogleNews,
        GoogleImages,
        GoogleFinance,
        GoogleJobs,
        GoogleLens,
        GoogleMaps,
        GoogleShopping,
    )

    # Expect problem_payload as (description, query)
    if isinstance(problem_payload, (list, tuple)) and len(problem_payload) >= 2:
        task_description, query = problem_payload[0], problem_payload[1]
    elif isinstance(problem_payload, str) and "|" in problem_payload:
        task_description, query = problem_payload.split("|", 1)
    else:
        raise ValueError(
            "Invalid problem payload for search agent; expected [desc, query]"
        )
    tools = [
        NodeTool(GoogleSearch),
        NodeTool(GoogleNews),
        NodeTool(GoogleImages),
        NodeTool(GoogleFinance),
        NodeTool(GoogleJobs),
        NodeTool(GoogleLens),
        NodeTool(GoogleMaps),
        NodeTool(GoogleShopping),
    ]
    return search_eval.build_search_agent(
        provider, model, tools, (task_description, query)
    )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified eval runner: single-run (tool-invoked) and full evaluations"
    )
    parser.add_argument(
        "--agent",
        required=True,
        choices=["math", "data", "browser", "search"],
        help="Agent type to run",
    )
    parser.add_argument(
        "--provider",
        required=False,
        type=str,
        help="Provider key (e.g., openai, gemini)",
    )
    parser.add_argument("--model", required=False, type=str, help="Model name")
    parser.add_argument(
        "--problem-json", required=False, type=str, help="JSON-encoded problem payload"
    )
    parser.add_argument(
        "--output-json",
        required=False,
        type=str,
        help="Output JSON path for result and usage",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full evaluation suite for the selected agent",
    )
    args = parser.parse_args()

    # Full evaluation mode
    if args.full:
        runner_path = os.path.abspath(__file__)
        if args.agent == "data":
            problems = data_eval.generate_iris_problems()
            evaluator = AgentEvaluator(
                models=MODELS,
                problems=problems,
                result_checker=data_eval.numeric_result_checker,
                concurrency=int(os.getenv("DATA_AGENT_CONCURRENCY", "8")),
                subprocess_runner_path=runner_path,
                subprocess_agent="data",
            )
            data_stats: Dict[str, ModelStats] = {m: ModelStats() for _, m in MODELS}
            data_logs: List[Any] = []
            from rich.live import Live
            from rich.console import Console

            console = Console()
            with Live(
                make_view(data_stats, data_logs, MODELS),
                refresh_per_second=8,
                console=console,
            ) as live:
                evaluator.on_update = lambda s, l: live.update(make_view(s, l, MODELS))  # type: ignore
                data_result: EvaluationResult = await evaluator.evaluate()
                live.update(make_view(data_result.stats, data_result.logs, MODELS))
            return

        if args.agent == "math":
            problems = math_eval.generate_math_problems()
            evaluator = AgentEvaluator(
                models=MODELS,
                problems=problems,
                result_checker=math_eval.numeric_result_checker,
                concurrency=int(os.getenv("MATH_AGENT_CONCURRENCY", "8")),
                subprocess_runner_path=runner_path,
                subprocess_agent="math",
            )
            math_stats: Dict[str, ModelStats] = {m: ModelStats() for _, m in MODELS}
            math_logs: List[Any] = []
            from rich.live import Live
            from rich.console import Console

            console = Console()
            with Live(
                make_view(math_stats, math_logs, MODELS),
                refresh_per_second=8,
                console=console,
            ) as live:
                evaluator.on_update = lambda s, l: live.update(make_view(s, l, MODELS))  # type: ignore
                math_result: EvaluationResult =  await evaluator.evaluate()
                live.update(make_view(math_result.stats, math_result.logs, MODELS))
            return

        if args.agent == "browser":
            tasks = browser_eval.generate_wikipedia_tasks()
            problems = [((desc, url), expected) for desc, url, expected in tasks]
            evaluator = AgentEvaluator(
                models=MODELS,
                problems=problems,
                result_checker=browser_eval.content_result_checker,
                concurrency=int(os.getenv("BROWSER_AGENT_CONCURRENCY", "4")),
                subprocess_runner_path=runner_path,
                subprocess_agent="browser",
            )
            browser_stats: Dict[str, ModelStats] = {m: ModelStats() for _, m in MODELS}
            browser_logs: List[Any] = []
            from rich.live import Live
            from rich.console import Console

            console = Console()
            # Browser agent uses different parameters for table formatting
            max_log_lines = int(os.getenv("BROWSER_AGENT_LOG_LINES", "50"))
            with Live(
                make_view(
                    browser_stats, browser_logs, MODELS, max_log_lines, "Task", True
                ),
                refresh_per_second=8,
                console=console,
            ) as live:
                evaluator.on_update = lambda s, l: live.update(make_view(s, l, MODELS, max_log_lines, "Task", True))  # type: ignore
                browser_result: EvaluationResult = await evaluator.evaluate()
                live.update(
                    make_view(
                        browser_result.stats,
                        browser_result.logs,
                        MODELS,
                        max_log_lines,
                        "Task",
                        True,
                    )
                )
            return

        if args.agent == "search":
            tasks = search_eval.generate_search_tasks()
            problems = [((desc, query), expected) for desc, query, expected in tasks]

            evaluator = AgentEvaluator(
                models=MODELS,
                problems=problems,
                result_checker=search_eval.content_result_checker,
                concurrency=int(os.getenv("SEARCH_AGENT_CONCURRENCY", "8")),
                subprocess_runner_path=runner_path,
                subprocess_agent="search",
            )
            search_stats: Dict[str, ModelStats] = {m: ModelStats() for _, m in MODELS}
            search_logs: List[Any] = []
            from rich.live import Live
            from rich.console import Console

            console = Console()
            max_log_lines = int(os.getenv("SEARCH_AGENT_LOG_LINES", "50"))
            with Live(
                make_view(
                    search_stats, search_logs, MODELS, max_log_lines, "Task", True
                ),
                refresh_per_second=8,
                console=console,
            ) as live:
                evaluator.on_update = lambda s, l: live.update(make_view(s, l, MODELS, max_log_lines, "Task", True))  # type: ignore
                search_result: EvaluationResult = await evaluator.evaluate()
                live.update(
                    make_view(
                        search_result.stats,
                        search_result.logs,
                        MODELS,
                        max_log_lines,
                        "Task",
                        True,
                    )
                )
            return

        raise SystemExit("Unknown agent type for full run")

    # Single-run mode (invoked by evaluator subprocess)
    if (
        not args.provider
        or not args.model
        or not args.problem_json
        or not args.output_json
    ):
        raise SystemExit(
            "Single-run mode requires --provider, --model, --problem-json, --output-json"
        )

    problem_payload = json.loads(args.problem_json)
    provider = await get_provider(args.provider)
    if args.agent == "math":
        agent = _build_math_agent(provider, args.model, problem_payload)
    elif args.agent == "data":
        agent = _build_data_agent(provider, args.model, problem_payload)
    elif args.agent == "browser":
        agent = _build_browser_agent(provider, args.model, problem_payload)
    elif args.agent == "search":
        agent = _build_search_agent(provider, args.model, problem_payload)
    else:
        raise SystemExit("Unknown agent type")
    await _execute_agent_once(agent, args.output_json)


if __name__ == "__main__":
    asyncio.run(main())
