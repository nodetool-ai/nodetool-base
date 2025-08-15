import argparse
import json
import os
import sys
import asyncio
from typing import Any, List, Dict

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


def default_provider_factory(provider_key: str) -> Any:
    from nodetool.chat.providers.base import ChatProvider
    from nodetool.chat.providers.openai_provider import OpenAIProvider
    from nodetool.chat.providers.gemini_provider import GeminiProvider
    from nodetool.chat.providers.anthropic_provider import AnthropicProvider
    from nodetool.chat.providers.huggingface_provider import HuggingFaceProvider

    if provider_key == "openai":
        return OpenAIProvider()
    elif provider_key == "gemini":
        return GeminiProvider()
    elif provider_key == "anthropic":
        return AnthropicProvider()
    elif provider_key.startswith("huggingface"):
        inference_provider = provider_key.split(":")[1]
        assert inference_provider in [
            "black-forest-labs",
            "cerebras",
            "cohere",
            "fal-ai",
            "featherless-ai",
            "fireworks-ai",
            "google",
            "mistral",
            "openai",
            "qwen",
            "together",
        ]
        return HuggingFaceProvider(inference_provider)  # type: ignore
    else:
        raise ValueError(f"Unknown provider key: {provider_key}")


def _build_math_agent(provider_key: str, model: str, problem_payload: Any) -> Any:
    from nodetool.agents.tools.node_tool import NodeTool
    from nodetool.nodes.lib.math import BinaryOp, UnaryOp

    provider = default_provider_factory(provider_key)
    tools = [NodeTool(BinaryOp), NodeTool(UnaryOp)]
    return math_eval.build_math_agent(provider, model, tools, problem_payload)


def _build_data_agent(provider_key: str, model: str, problem_payload: Any) -> Any:
    provider = default_provider_factory(provider_key)
    # Reuse exported tools and builder from data eval
    return data_eval.build_data_agent(provider, model, data_eval.tools, problem_payload)


def _build_browser_agent(provider_key: str, model: str, problem_payload: Any) -> Any:
    from nodetool.agents.tools.browser_tools import AgenticBrowserTool, BrowserTool

    provider = default_provider_factory(provider_key)
    # Expect problem_payload as (description, url)
    if isinstance(problem_payload, (list, tuple)) and len(problem_payload) >= 2:
        task_description, url = problem_payload[0], problem_payload[1]
    elif isinstance(problem_payload, str) and "|" in problem_payload:
        task_description, url = problem_payload.split("|", 1)
    else:
        raise ValueError(
            "Invalid problem payload for browser agent; expected [desc, url]"
        )
    difficulty = os.getenv("BROWSER_AGENT_DIFFICULTY", "hard").strip().lower()
    tools = [AgenticBrowserTool()] if difficulty == "hard" else [BrowserTool()]
    return browser_eval.build_browser_agent(
        provider, model, tools, (task_description, url)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified eval runner: single-run (tool-invoked) and full evaluations"
    )
    parser.add_argument(
        "--agent",
        required=True,
        choices=["math", "data", "browser"],
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
            models = data_eval.MODELS
            tools = data_eval.tools
            evaluator = AgentEvaluator(
                models=models,
                problems=problems,
                result_checker=data_eval.numeric_result_checker,
                tools=tools,
                concurrency=int(os.getenv("DATA_AGENT_CONCURRENCY", "8")),
                subprocess_runner_path=runner_path,
                subprocess_agent="data",
            )
            data_stats: Dict[str, ModelStats] = {m: ModelStats() for _, m in models}
            data_logs: List[Any] = []
            from rich.live import Live
            from rich.console import Console

            console = Console()
            with Live(
                data_eval.make_view(data_stats, data_logs),
                refresh_per_second=8,
                console=console,
            ) as live:
                evaluator.on_update = lambda s, l: live.update(data_eval.make_view(s, l))  # type: ignore
                data_result: EvaluationResult = (
                    asyncio.get_event_loop().run_until_complete(evaluator.evaluate())
                )
                live.update(data_eval.make_view(data_result.stats, data_result.logs))
            return

        if args.agent == "math":
            problems = math_eval.generate_math_problems()
            models = math_eval.MODELS
            from nodetool.agents.tools.node_tool import NodeTool
            from nodetool.nodes.lib.math import BinaryOp, UnaryOp

            tools = [NodeTool(BinaryOp), NodeTool(UnaryOp)]
            evaluator = AgentEvaluator(
                models=models,
                problems=problems,
                result_checker=math_eval.numeric_result_checker,
                tools=tools,
                concurrency=int(os.getenv("MATH_AGENT_CONCURRENCY", "8")),
                subprocess_runner_path=runner_path,
                subprocess_agent="math",
            )
            math_stats: Dict[str, ModelStats] = {m: ModelStats() for _, m in models}
            math_logs: List[Any] = []
            from rich.live import Live
            from rich.console import Console

            console = Console()
            with Live(
                math_eval.make_view(math_stats, math_logs),
                refresh_per_second=8,
                console=console,
            ) as live:
                evaluator.on_update = lambda s, l: live.update(math_eval.make_view(s, l))  # type: ignore
                math_result: EvaluationResult = (
                    asyncio.get_event_loop().run_until_complete(evaluator.evaluate())
                )
                live.update(math_eval.make_view(math_result.stats, math_result.logs))
            return

        if args.agent == "browser":
            tasks = browser_eval.generate_wikipedia_tasks()
            problems = [((desc, url), expected) for desc, url, expected in tasks]
            models = browser_eval.MODELS
            from nodetool.agents.tools.browser_tools import AgenticBrowserTool

            tools = [AgenticBrowserTool()]
            evaluator = AgentEvaluator(
                models=models,
                problems=problems,
                result_checker=browser_eval.content_result_checker,
                tools=tools,
                concurrency=int(os.getenv("BROWSER_AGENT_CONCURRENCY", "4")),
                subprocess_runner_path=runner_path,
                subprocess_agent="browser",
            )
            browser_stats: Dict[str, ModelStats] = {m: ModelStats() for _, m in models}
            browser_logs: List[Any] = []
            from rich.live import Live
            from rich.console import Console

            console = Console()
            with Live(
                browser_eval.make_view(browser_stats, browser_logs),
                refresh_per_second=8,
                console=console,
            ) as live:
                evaluator.on_update = lambda s, l: live.update(browser_eval.make_view(s, l))  # type: ignore
                browser_result: EvaluationResult = (
                    asyncio.get_event_loop().run_until_complete(evaluator.evaluate())
                )
                live.update(
                    browser_eval.make_view(browser_result.stats, browser_result.logs)
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
    if args.agent == "math":
        agent = _build_math_agent(args.provider, args.model, problem_payload)
    elif args.agent == "data":
        agent = _build_data_agent(args.provider, args.model, problem_payload)
    elif args.agent == "browser":
        agent = _build_browser_agent(args.provider, args.model, problem_payload)
    else:
        raise SystemExit("Unknown agent type")
    asyncio.run(_execute_agent_once(agent, args.output_json))


if __name__ == "__main__":
    main()
