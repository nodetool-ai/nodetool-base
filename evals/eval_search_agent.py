import os
from typing import Any, List, Sequence, Tuple

from nodetool.agents.simple_agent import SimpleAgent


# Models to evaluate with. Keep consistent with other evals.
MODELS: List[Tuple[str, str]] = [
    ("openai", "gpt-5-mini"),
    ("gemini", "gemini-2.5-flash-lite"),
    ("anthropic", "claude-3-5-haiku-20241022"),
    ("huggingface:cerebras", "openai/gpt-oss-120b"),
]


# Keep a modest rolling buffer of recent agent results in the log panel
MAX_LOG_LINES: int = int(os.getenv("SEARCH_AGENT_LOG_LINES", "50"))


def generate_search_tasks(difficulty: str | None = None) -> List[Tuple[str, str, str]]:
    """Return a suite of richer search tasks with expected results.

    Task tuple format: (description, query, expected_substring)

    Notes:
    - Tasks are designed to encourage usage of GoogleSearch, GoogleNews, GoogleImages,
      GoogleFinance, GoogleJobs, GoogleLens, GoogleMaps, GoogleShopping.
    - Expected substrings are chosen to be relatively robust and case-insensitive.
    """
    return [
        # GoogleSearch (general web)
        (
            "Identify the programming language known for the 'import this' easter egg",
            "language with 'import this' easter egg",
            "python",
        ),
        # GoogleNews (recent articles)
        (
            "Find a recent news headline that mentions the James Webb Space Telescope",
            "James Webb Space Telescope latest news",
            "james webb",
        ),
        # GoogleImages (keyword-based image search)
        (
            "Using image search, determine the animal featured in the Mozilla Firefox logo",
            "Mozilla Firefox logo animal",
            "fox",
        ),
        # GoogleImages (reverse image search)
        (
            "Identify the painting from this image URL (reverse image)",
            "https://upload.wikimedia.org/wikipedia/commons/6/6a/Mona_Lisa.jpg",
            "mona lisa",
        ),
        # GoogleFinance
        (
            "Retrieve the stock ticker symbol for Tesla, Inc.",
            "Tesla stock ticker",
            "tsla",
        ),
        # GoogleJobs
        (
            "Find job listings for a Data Scientist role in New York",
            "Data Scientist New York jobs",
            "new york",
        ),
        # GoogleLens (visual match)
        (
            "Identify the landmark shown in this image URL (visual search)",
            "https://upload.wikimedia.org/wikipedia/commons/a/af/Tour_Eiffel_Wikimedia_Commons.jpg",
            "eiffel",
        ),
        # GoogleMaps
        (
            "Find the street address of the Space Needle in Seattle",
            "Space Needle address",
            "400 broad st",
        ),
        # GoogleShopping
        (
            "Find a shopping listing for Apple AirPods Pro (2nd generation)",
            "Apple AirPods Pro (2nd generation) shopping",
            "airpods pro",
        ),
        # Bonus Maps/Local info
        (
            "Find the city of the Louvre Museum",
            "Louvre Museum location",
            "paris",
        ),
    ]


def build_objective(task_description: str, query: str) -> str:
    """Build the objective for the search agent."""
    return f"""
        Use the provided SERP tools (Google Search, News, Images, Finance, Jobs, Lens, Maps, Shopping)
        to perform the search and extract the requested information. Prefer concise, factual answers grounded in
        the search results. Choose tools based on the query:
        - If the query includes an image URL, prefer Google Images (reverse) or Google Lens.
        - If the query asks about places or addresses, use Google Maps.
        - For stock or ticker information, use Google Finance.
        - For job listings, use Google Jobs.
        - For product listings, use Google Shopping.
        - For current events, use Google News.
        You may combine multiple tools when helpful.

        Task: {task_description}
        Query: {query}

        Return the extracted information in the 'content' field as a concise answer.
    """


def build_search_agent(
    provider: Any, model: str, tools: Sequence[Any], problem: Any
) -> Any:
    # Extract task_description and query from the problem
    if isinstance(problem, str) and "|" in problem:
        task_description, query = problem.split("|", 1)
    elif isinstance(problem, (tuple, list)) and len(problem) >= 2:
        task_description, query = problem[0], problem[1]
    else:
        raise ValueError(f"Invalid problem format: {problem}")

    return SimpleAgent(
        name="Search Agent",
        objective=build_objective(task_description, query),
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
        return expected_lower in content

    except Exception as e:
        print(f"Error in content_result_checker: {e}")
        return False
