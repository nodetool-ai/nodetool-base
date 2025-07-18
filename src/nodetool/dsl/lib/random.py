from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class RandomBool(GraphNode):
    """
    Return a random boolean value.
    random, boolean, coinflip, bool

    Use cases:
    - Make random yes/no decisions
    - Simulate coin flips
    - Introduce randomness in control flow
    """

    @classmethod
    def get_node_type(cls):
        return "lib.random.RandomBool"


class RandomChoice(GraphNode):
    """
    Select a random element from a list.
    random, choice, select, pick

    Use cases:
    - Choose a random sample from options
    - Implement simple lottery behaviour
    - Pick a random item from user input
    """

    options: list[Any] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="List of options"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.random.RandomChoice"


class RandomFloat(GraphNode):
    """
    Generate a random floating point number within a range.
    random, float, number, rand, uniform

    Use cases:
    - Create random probabilities
    - Generate noisy data for testing
    - Produce random values for simulations
    """

    minimum: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.0, description="Minimum value"
    )
    maximum: float | GraphNode | tuple[GraphNode, str] = Field(
        default=1.0, description="Maximum value"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.random.RandomFloat"


class RandomInt(GraphNode):
    """
    Generate a random integer within a range.
    random, integer, number, rand, randint

    Use cases:
    - Pick a random index or identifier
    - Create randomized counters or IDs
    - Sample integers for testing
    """

    minimum: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="Minimum value (inclusive)"
    )
    maximum: int | GraphNode | tuple[GraphNode, str] = Field(
        default=100, description="Maximum value (inclusive)"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.random.RandomInt"
