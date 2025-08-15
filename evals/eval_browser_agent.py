import asyncio
import os
import sys
import json
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from rich.table import Table
from rich.columns import Columns
from rich.console import Console
from rich.live import Live

from nodetool.agents.simple_agent import SimpleAgent
from nodetool.agents.agent_evaluator import (
    ModelStats,
)


MODELS: List[Tuple[str, str]] = [
    # ("openai", "gpt-5"),
    ("openai", "gpt-5-mini"),
    # ("gemini", "gemini-2.5-flash"),
    ("gemini", "gemini-2.5-flash-lite"),
    # ("anthropic", "claude-sonnet-4-20250514"),
    ("anthropic", "claude-3-5-haiku-20241022"),
    ("huggingface:cerebras", "openai/gpt-oss-120b"),
    # ("huggingface:cerebras", "Qwen/Qwen3-Coder-480B-A35B-Instruct"),
]


# Keep a modest rolling buffer of recent agent results in the log panel
MAX_LOG_LINES: int = int(os.getenv("BROWSER_AGENT_LOG_LINES", "50"))


def generate_wikipedia_tasks(
    difficulty: str | None = None,
) -> List[Tuple[str, str, str]]:
    """Return a suite of Wikipedia tasks of varying difficulty with expected results.

    Task tuple format: (description, starting_url, expected_substring)
    """
    tasks: List[Tuple[str, str, str]] = []

    # Basic article information extraction
    return [
        (
            "Find the birth year of Albert Einstein from his Wikipedia page",
            "https://en.wikipedia.org/wiki/Albert_Einstein",
            "1879",
        ),
        (
            "Extract the year the World Wide Web was invented from its Wikipedia page",
            "https://en.wikipedia.org/wiki/World_Wide_Web",
            "1989",
        ),
        (
            "Find the number of bones in an adult human body from the Wikipedia article on human skeleton",
            "https://en.wikipedia.org/wiki/Human_skeleton",
            "206",
        ),
        (
            "Extract the speed of light value from its Wikipedia page",
            "https://en.wikipedia.org/wiki/Speed_of_light",
            "299,792,458",
        ),
        (
            "From the Python (programming language) page, find the year Python 3.0 was released",
            "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "2008",
        ),
        (
            "From the Apollo 11 page, identify the name of the command module pilot",
            "https://en.wikipedia.org/wiki/Apollo_11",
            "Michael Collins",
        ),
        (
            "From the Traveling salesman problem page, state the complexity class of the decision version",
            "https://en.wikipedia.org/wiki/Travelling_salesman_problem",
            "NP-complete",
        ),
        (
            "From the Fibonacci number page, provide the name of the closed-form expression for the nth Fibonacci number",
            "https://en.wikipedia.org/wiki/Fibonacci_number",
            "Binet",
        ),
        (
            "Starting from the Alan Turing page, follow the link to the Turing Award page and report the year it was first awarded",
            "https://en.wikipedia.org/wiki/Alan_Turing",
            "1966",
        ),
        (
            "From the Graph theory page, navigate to the Seven Bridges of Königsberg and provide the surname of the mathematician who solved it",
            "https://en.wikipedia.org/wiki/Graph_theory",
            "Euler",
        ),
        (
            "Starting at the RSA (cryptosystem) page, navigate to Shor's algorithm and state the year it was proposed",
            "https://en.wikipedia.org/wiki/RSA_(cryptosystem)",
            "1994",
        ),
        (
            "From the World Wide Web page, follow the HTTP page and report the default port number used by HTTP",
            "https://en.wikipedia.org/wiki/World_Wide_Web",
            "80",
        ),
        (
            "Starting on the Python (programming language) page, navigate to Guido van Rossum and report his country of birth",
            "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "Netherlands",
        ),
        (
            "From the Haskell (programming language) page, open the Monad (functional programming) page and name the researcher who popularized monads in Haskell",
            "https://en.wikipedia.org/wiki/Haskell_(programming_language)",
            "Philip Wadler",
        ),
        (
            "From the Prime number page, navigate to the Riemann hypothesis and provide the year Riemann published the paper introducing it",
            "https://en.wikipedia.org/wiki/Prime_number",
            "1859",
        ),
        (
            "From the United States page, go to the Constitution of the United States and report the total number of amendments",
            "https://en.wikipedia.org/wiki/United_States",
            "27",
        ),
        (
            "From the Nobel Prize page, navigate to Marie Curie and report how many Nobel Prizes she received",
            "https://en.wikipedia.org/wiki/Nobel_Prize",
            "2",
        ),
        (
            "From the Artificial intelligence page, navigate to the Turing test and provide the surname of the person who proposed it",
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "Turing",
        ),
    ]


def build_objective(task_description: str, url: str) -> str:
    """Build the objective for the browser agent."""
    return f"""
        Use the provided browser tools to open the starting URL and extract the requested information.
        You may navigate within and across Wikipedia pages by following links, searching within pages, and reading tables.
        Be precise and return only the requested fact(s) as a short answer.

        Task: {task_description}
        Start URL: {url}

        Return the extracted information in the 'content' field as a concise answer.
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
    table.add_column("Task")
    table.add_column("Result", justify="right")
    table.add_column("Correct", justify="center")
    table.add_column("Runtime (s)", justify="right")
    if not log_entries:
        return table
    for entry in log_entries[-MAX_LOG_LINES:]:
        status = (
            "✓" if entry.correct is True else ("✗" if entry.correct is False else "—")
        )
        result_text = (
            "None"
            if entry.result is None
            else (
                str(entry.result)[:50] + "..."
                if len(str(entry.result)) > 50
                else str(entry.result)
            )
        )
        table.add_row(
            entry.model,
            entry.problem[:30] + "..." if len(entry.problem) > 30 else entry.problem,
            result_text,
            status,
            f"{entry.runtime_seconds:.2f}",
        )
    return table


def make_view(stats: Dict[str, ModelStats], log_entries: List[Any]):
    stats_table = make_table(stats)
    logs_table = make_log_table(log_entries)
    return Columns([stats_table, logs_table], equal=True, expand=True)


def build_browser_agent(
    provider: Any, model: str, tools: Sequence[Any], problem: Any
) -> Any:
    # Extract task_description and url from the problem
    if isinstance(problem, str) and "|" in problem:
        task_description, url = problem.split("|", 1)
    elif isinstance(problem, (tuple, list)) and len(problem) >= 2:
        task_description, url = problem[0], problem[1]
    else:
        raise ValueError(f"Invalid problem format: {problem}")

    return SimpleAgent(
        name="Browser Agent",
        objective=build_objective(task_description, url),
        provider=provider,
        model=model,
        tools=list(tools),
        output_schema={"content": "string"},
    )


def content_result_checker(result: Any, expected: Any) -> bool:
    """Check if the extracted content contains the expected information."""
    try:
        if expected is None:
            return result is not None

        content: str = ""
        if isinstance(result, dict) and "content" in result:
            content = str(result["content"]).lower()
        else:
            content = str(result).lower()

        if not content:
            return False

        expected_lower = str(expected).lower()

        # Check if expected content is found in the result
        return expected_lower in content

    except Exception as e:
        print(f"Error in content_result_checker: {e}")
        return False
