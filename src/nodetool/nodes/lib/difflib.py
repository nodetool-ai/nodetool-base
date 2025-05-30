from difflib import SequenceMatcher, get_close_matches, unified_diff
from pydantic import Field
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class SimilarityRatio(BaseNode):
    """
    Calculates the similarity ratio between two strings.
    difflib, similarity, ratio, compare

    Use cases:
    - Fuzzy string matching
    - Compare document versions
    - Evaluate similarity of user input
    """

    a: str = Field(default="", description="First string to compare")
    b: str = Field(default="", description="Second string to compare")

    @classmethod
    def get_title(cls):
        return "Sequence Similarity Ratio"

    async def process(self, context: ProcessingContext) -> float:
        return SequenceMatcher(None, self.a, self.b).ratio()


class GetCloseMatches(BaseNode):
    """
    Finds close matches for a word within a list of possibilities.
    difflib, fuzzy, match

    Use cases:
    - Suggest alternatives for misspelled words
    - Map user input to valid options
    - Provide recommendations based on partial text
    """

    word: str = Field(default="", description="Word to match")
    possibilities: list[str] = Field(default=[], description="List of possible words")
    n: int = Field(default=3, ge=1, description="Maximum number of matches to return")
    cutoff: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Minimum similarity ratio"
    )

    @classmethod
    def get_title(cls):
        return "Get Close Matches"

    async def process(self, context: ProcessingContext) -> list[str]:
        return get_close_matches(
            self.word, self.possibilities, n=self.n, cutoff=self.cutoff
        )


class UnifiedDiff(BaseNode):
    """
    Generates a unified diff between two texts.
    difflib, diff, compare

    Use cases:
    - Display differences between versions of text files
    - Highlight changes in user submitted documents
    - Compare code snippets
    """

    a: str = Field(default="", description="Original text")
    b: str = Field(default="", description="Modified text")
    fromfile: str = Field(default="a", description="Name of the original file")
    tofile: str = Field(default="b", description="Name of the modified file")
    lineterm: str = Field(default="\n", description="Line terminator")

    @classmethod
    def get_title(cls):
        return "Unified Diff"

    async def process(self, context: ProcessingContext) -> str:
        diff_lines = unified_diff(
            self.a.splitlines(),
            self.b.splitlines(),
            fromfile=self.fromfile,
            tofile=self.tofile,
            lineterm=self.lineterm,
        )
        return "\n".join(diff_lines)
