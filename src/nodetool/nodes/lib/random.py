import random
from typing import Any
from pydantic import Field
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class RandomInt(BaseNode):
    """
    Generate a random integer within a range.
    random, integer, number, rand, randint

    Use cases:
    - Pick a random index or identifier
    - Create randomized counters or IDs
    - Sample integers for testing
    """

    minimum: int = Field(default=0, description="Minimum value (inclusive)")
    maximum: int = Field(default=100, description="Maximum value (inclusive)")

    _expose_as_tool: bool = True

    async def process(self, context: ProcessingContext) -> int:
        return random.randint(self.minimum, self.maximum)


class RandomFloat(BaseNode):
    """
    Generate a random floating point number within a range.
    random, float, number, rand, uniform

    Use cases:
    - Create random probabilities
    - Generate noisy data for testing
    - Produce random values for simulations
    """

    minimum: float = Field(default=0.0, description="Minimum value")
    maximum: float = Field(default=1.0, description="Maximum value")

    _expose_as_tool: bool = True

    async def process(self, context: ProcessingContext) -> float:
        return random.uniform(self.minimum, self.maximum)


class RandomChoice(BaseNode):
    """
    Select a random element from a list.
    random, choice, select, pick

    Use cases:
    - Choose a random sample from options
    - Implement simple lottery behaviour
    - Pick a random item from user input
    """

    options: list[Any] = Field(default=[], description="List of options")

    _expose_as_tool: bool = True

    async def process(self, context: ProcessingContext) -> Any:
        if not self.options:
            raise ValueError("options list is empty")
        return random.choice(self.options)


class RandomBool(BaseNode):
    """
    Return a random boolean value.
    random, boolean, coinflip, bool

    Use cases:
    - Make random yes/no decisions
    - Simulate coin flips
    - Introduce randomness in control flow
    """

    _expose_as_tool: bool = True

    async def process(self, context: ProcessingContext) -> bool:
        return random.choice([True, False])
