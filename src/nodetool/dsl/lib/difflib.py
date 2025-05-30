from pydantic import Field
from nodetool.dsl.graph import GraphNode


class GetCloseMatches(GraphNode):
    """
    Finds close matches for a word within a list of possibilities.
    difflib, fuzzy, match

    Use cases:
    - Suggest alternatives for misspelled words
    - Map user input to valid options
    - Provide recommendations based on partial text
    """

    word: str | GraphNode | tuple[GraphNode, str] = Field(default='', description='Word to match')
    possibilities: list[str] | GraphNode | tuple[GraphNode, str] = Field(default=[], description='List of possible words')
    n: int | GraphNode | tuple[GraphNode, str] = Field(default=3, description='Maximum number of matches to return')
    cutoff: float | GraphNode | tuple[GraphNode, str] = Field(default=0.6, description='Minimum similarity ratio')

    @classmethod
    def get_node_type(cls): return "lib.difflib.GetCloseMatches"



class SimilarityRatio(GraphNode):
    """
    Calculates the similarity ratio between two strings.
    difflib, similarity, ratio, compare

    Use cases:
    - Fuzzy string matching
    - Compare document versions
    - Evaluate similarity of user input
    """

    a: str | GraphNode | tuple[GraphNode, str] = Field(default='', description='First string to compare')
    b: str | GraphNode | tuple[GraphNode, str] = Field(default='', description='Second string to compare')

    @classmethod
    def get_node_type(cls): return "lib.difflib.SimilarityRatio"



class UnifiedDiff(GraphNode):
    """
    Generates a unified diff between two texts.
    difflib, diff, compare

    Use cases:
    - Display differences between versions of text files
    - Highlight changes in user submitted documents
    - Compare code snippets
    """

    a: str | GraphNode | tuple[GraphNode, str] = Field(default='', description='Original text')
    b: str | GraphNode | tuple[GraphNode, str] = Field(default='', description='Modified text')
    fromfile: str | GraphNode | tuple[GraphNode, str] = Field(default='a', description='Name of the original file')
    tofile: str | GraphNode | tuple[GraphNode, str] = Field(default='b', description='Name of the modified file')
    lineterm: str | GraphNode | tuple[GraphNode, str] = Field(default='\n', description='Line terminator')

    @classmethod
    def get_node_type(cls): return "lib.difflib.UnifiedDiff"


