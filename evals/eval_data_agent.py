import asyncio
import json
import math
import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.table import Table

from nodetool.agents.agent_evaluator import (
    AgentEvaluator,
    EvaluationResult,
    ModelStats,
)
from nodetool.agents.simple_agent import SimpleAgent
from nodetool.agents.tools.node_tool import NodeTool
from nodetool.chat.providers.base import ChatProvider
from nodetool.nodes.nodetool.data import (
    AddColumn,
    Aggregate,
    Append,
    Filter,
    DropDuplicates,
    DropNA,
    ExtractColumn,
    FillNA,
    FindRow,
    Join,
    LoadCSVFile,
    Merge,
    Pivot,
    Rename,
    SelectColumn,
    Slice,
    SortByColumn,
    ToList,
)


# IRIS-based evaluator for the Data Agent.
# Note: Some data tools exist but are not included here because they require local files or asset folders:
# - LoadCSVFile, LoadCSVAssets, SaveDataframe, ImportCSV, FromList, JSONToDataframe, RowIterator.
# This IRIS-focused eval covers the tools listed in `nodes` below.


MODELS: List[Tuple[str, str]] = [
    ("openai", "gpt-5"),
    ("openai", "gpt-5-mini"),
    ("gemini", "gemini-2.5-pro"),
    ("gemini", "gemini-2.5-flash"),
    ("anthropic", "claude-sonnet-4-20250514"),
    ("anthropic", "claude-3-5-haiku-20241022"),
]


IRIS_FILE = os.path.join(os.path.dirname(__file__), "iris.csv")


def build_objective(task_description: str, data: dict[str, str]) -> str:
    return f"""
        You are a data analyst with the ability to use tools to analyze data.
        Use ONLY the available tools to load and manipulate the data to compute the answer.
        Return the final numeric result in the 'value' field.
        Load the data tables from the file paths provided.

        Task: {task_description}
        Data tables (name → file path): {json.dumps(data)}
    """


def build_data_agent(
    provider: ChatProvider,
    model: str,
    tools: Sequence[Any],
    task_description: str,
) -> Any:
    return SimpleAgent(
        name="Data Agent",
        objective=build_objective(
            task_description,
            {
                "iris": IRIS_FILE,
            },
        ),
        provider=provider,
        model=model,
        tools=list(tools),
        output_schema={"value": "number"},
    )


nodes = [
    LoadCSVFile,
    Filter,
    Aggregate,
    Join,
    Pivot,
    SortByColumn,
    Append,
    AddColumn,
    DropDuplicates,
    ExtractColumn,
    DropNA,
    FillNA,
    FindRow,
    Merge,
    SelectColumn,
    Slice,
    Rename,
    ToList,
]

tools = [NodeTool(node) for node in nodes]


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


def load_iris_df() -> pd.DataFrame:
    return pd.read_csv(IRIS_FILE)


def generate_iris_problems() -> List[Tuple[str, float]]:
    """Create IRIS tasks with deterministic numeric answers.

    Each task is phrased to naturally encourage the use of one or more tools in `nodes`.
    """
    df = load_iris_df()
    problems: List[Tuple[str, float]] = []

    # 1) Aggregate (mean) – sepal.length
    problems.append(
        (
            "Compute the mean of the sepal.length over the entire IRIS dataset.",
            float(df["sepal.length"].mean()),
        )
    )

    # 2) Filter + count – petal.width > 1.5
    problems.append(
        (
            "How many rows have petal.width strictly greater than 1.5?",
            float((df["petal.width"] > 1.5).sum()),
        )
    )

    # 3) DropDuplicates on variety classes
    problems.append(
        (
            "Count the number of unique variety classes using a duplicates removal approach.",
            float(df["variety"].nunique()),
        )
    )

    # 4) SortByColumn + Slice: 11th smallest sepal.width -> sepal.length value
    sorted_df = df.sort_values("sepal.width").reset_index(drop=True)
    problems.append(
        (
            "After sorting by sepal.width ascending, what is the sepal.length at row index 10?",
            float(sorted_df.loc[10, "sepal.length"]),
        )
    )

    # 5) Append two slices: first 50 + next 50 → total rows
    problems.append(
        (
            "Append the first 50 rows with the next 50 rows and report the number of rows in the result.",
            100.0,
        )
    )

    # 6) Merge two single-column dataframes horizontally, then report the number of columns
    problems.append(
        (
            "Merge horizontally a dataframe with only sepal.length and another with only sepal.width, then return the number of columns.",
            2.0,
        )
    )

    # 7) AddColumn + FillNA on a small slice: sum of filled values
    problems.append(
        (
            "Take the first 5 rows, add a new column named tmp with values [1, null, 3, null, 5], fill missing values in tmp with 0, then return the sum of tmp.",
            9.0,
        )
    )

    # 8) AddColumn + DropNA on a small slice: remaining row count
    problems.append(
        (
            "Take the first 5 rows, add a new column named tmp with values [1, null, 3, null, 5], drop rows with NA, then return the number of remaining rows.",
            3.0,
        )
    )

    # 9) FindRow: first row with variety == 'Virginica' and petal.length > 6 → return its sepal.length
    mask = (df["variety"] == "Virginica") & (df["petal.length"] > 6)
    matched_indices = df.index[mask].tolist()
    if len(matched_indices) == 0:
        # Fallback to a slightly looser condition if rare
        mask2 = (df["variety"] == "Virginica") & (df["petal.length"] > 5.8)
        matched_indices = df.index[mask2].tolist()
    first_idx = int(matched_indices[0])
    problems.append(
        (
            "Find the first row where variety == 'Virginica' and petal.length > 6, then return its sepal.length.",
            float(df.loc[int(first_idx), "sepal.length"]),
        )
    )

    # 10) Pivot: index=variety, columns=variety, values=petal.length, agg=mean → sum diagonal (means)
    means_by_variety = df.groupby("variety")["petal.length"].mean().sort_index()
    problems.append(
        (
            "Create a pivot with index variety, columns variety, values petal.length using mean aggregation, and return the sum of the diagonal values.",
            float(means_by_variety.sum()),
        )
    )

    # 11) Join on identical column sets: unique variety vs unique variety → row count 3
    problems.append(
        (
            "Select only the variety column, drop duplicates to get unique classes in two separate dataframes, join them on variety, then return the number of rows.",
            3.0,
        )
    )

    # 12) Rename a column then aggregate: mean of sepal_length after renaming
    problems.append(
        (
            "Rename column 'sepal.length' to 'sepal_length' and return the mean of 'sepal_length'.",
            float(df["sepal.length"].mean()),
        )
    )

    # 13) ExtractColumn + ToList: sum of the first 10 sepal.width values
    problems.append(
        (
            "From the first 10 rows, extract the 'sepal.width' column as a list and return the sum of the list.",
            float(df["sepal.width"].iloc[:10].sum()),
        )
    )

    # 14) SelectColumn then Aggregate: mean petal.width for variety=='Versicolor'
    mean_pw_versicolor = df[df["variety"] == "Versicolor"]["petal.width"].mean()
    problems.append(
        (
            "Filter rows where variety == 'Versicolor', select the 'petal.width' column, and return its mean.",
            float(mean_pw_versicolor),
        )
    )

    return problems


def make_table(stats: Dict[str, ModelStats]) -> Table:
    table = Table()
    table.add_column("Model")
    table.add_column("Finished Tests", justify="right")
    table.add_column("Correct Results", justify="right")
    table.add_column("Input Tokens", justify="right")
    table.add_column("Output Tokens", justify="right")
    table.add_column("Avg Runtime (s)", justify="right")
    for _, model in MODELS:
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
    for entry in log_entries[-50:]:
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
    return Columns(
        [make_table(stats), make_log_table(log_entries)], equal=True, expand=True
    )


async def main():
    problems = generate_iris_problems()
    concurrency = int(os.getenv("DATA_AGENT_CONCURRENCY", "8"))

    evaluator = AgentEvaluator(
        models=MODELS,
        problems=problems,
        build_agent_fn=build_data_agent,
        result_checker=numeric_result_checker,
        tools=tools,
        concurrency=concurrency,
    )

    stats: Dict[str, ModelStats] = {m: ModelStats() for _, m in MODELS}
    logs: List[Any] = []
    console = Console()
    with Live(make_view(stats, logs), refresh_per_second=8, console=console) as live:
        evaluator.on_update = lambda s, l: live.update(make_view(s, l))  # type: ignore
        result: EvaluationResult = await evaluator.evaluate()
        live.update(make_view(result.stats, result.logs))


if __name__ == "__main__":
    asyncio.run(main())
