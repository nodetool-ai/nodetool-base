from enum import Enum
from typing import Any

from pydantic import Field

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class ConditionalSwitch(BaseNode):
    """
    Return one of two values based on boolean condition.

    Evaluates condition and returns if_true value when condition is true, otherwise
    returns if_false value. Simple inline conditional without branching the workflow.

    Parameters:
    - condition (required, default=False): Boolean to evaluate
    - if_true (optional, default=()): Value returned when condition is true
    - if_false (optional, default=()): Value returned when condition is false

    Returns: Either if_true or if_false value

    Typical usage: Select between two alternatives, implement ternary logic, or
    choose values without branching workflow. For workflow branching use If node
    instead. Follow with further processing or output.

    if, condition, flow-control, branch, true, false, switch, toggle
    """

    condition: bool = Field(default=False, description="The condition to check")
    if_true: Any = Field(
        default=(), description="The value to return if the condition is true"
    )
    if_false: Any = Field(
        default=(), description="The value to return if the condition is false"
    )

    async def process(self, context: ProcessingContext) -> Any:
        return self.if_true if self.condition else self.if_false


class LogicalOperator(BaseNode):
    """
    Combine two boolean values using logical operations (AND, OR, XOR, NAND, NOR).

    Applies selected logical operation to two boolean inputs and returns the result.
    Supports standard boolean algebra operations.

    Parameters:
    - a (required, default=False): First boolean operand
    - b (required, default=False): Second boolean operand
    - operation (required, default=AND): Logical operation (and, or, xor, nand, nor)

    Returns: Boolean result of the operation

    Typical usage: Combine multiple conditions, implement complex logic rules, or
    create compound filters. Precede with comparison nodes. Follow with If node for
    branching or further boolean logic.

    boolean, logic, operator, condition, flow-control, branch, else, true, false, switch, toggle
    """

    class BooleanOperation(str, Enum):
        AND = "and"
        OR = "or"
        XOR = "xor"
        NAND = "nand"
        NOR = "nor"

    a: bool = Field(default=False, description="First boolean input")
    b: bool = Field(default=False, description="Second boolean input")
    operation: BooleanOperation = Field(
        default=BooleanOperation.AND, description="Logical operation to perform"
    )

    async def process(self, context: ProcessingContext) -> bool:
        if self.operation == self.BooleanOperation.AND:
            return self.a and self.b
        elif self.operation == self.BooleanOperation.OR:
            return self.a or self.b
        elif self.operation == self.BooleanOperation.XOR:
            return self.a ^ self.b
        elif self.operation == self.BooleanOperation.NAND:
            return not (self.a and self.b)
        elif self.operation == self.BooleanOperation.NOR:
            return not (self.a or self.b)
        else:
            raise ValueError(f"Unsupported operation: {self.operation}")


class Not(BaseNode):
    """
    Invert boolean value (logical NOT).

    Returns the opposite of the input boolean. True becomes False, False becomes True.

    Parameters:
    - value (required, default=False): Boolean to negate

    Returns: Inverted boolean

    Typical usage: Invert conditions, implement toggle logic, or create opposite
    branches. Follow with If node or other boolean operations.

    boolean, logic, not, invert, !, negation, condition, else, true, false, switch, toggle, flow-control, branch
    """

    value: bool = Field(default=False, description="Boolean input to negate")

    async def process(self, context: ProcessingContext) -> bool:
        return not self.value


class Compare(BaseNode):
    """
    Compare two numeric values using specified comparison operator.

    Evaluates comparison between two numbers and returns boolean result. Supports
    equality, inequality, and relational comparisons.

    Parameters:
    - a (required, default=0): First value (int or float)
    - b (required, default=0): Second value (int or float)
    - comparison (required, default==): Operator (==, !=, >, <, >=, <=)

    Returns: Boolean result of comparison

    Typical usage: Create conditional logic based on numeric values, implement
    thresholds, or filter data. Follow with If node for branching or boolean
    operations for complex conditions.

    compare, condition, logic
    """

    class Comparison(str, Enum):
        EQUAL = "=="
        NOT_EQUAL = "!="
        GREATER_THAN = ">"
        LESS_THAN = "<"
        GREATER_THAN_OR_EQUAL = ">="
        LESS_THAN_OR_EQUAL = "<="

    a: int | float = Field(default=0, description="First value to compare")
    b: int | float = Field(default=0, description="Second value to compare")
    comparison: Comparison = Field(
        default=Comparison.EQUAL, description="Comparison operator to use"
    )

    async def process(self, context: ProcessingContext) -> bool:
        if self.comparison == self.Comparison.EQUAL:
            return self.a == self.b
        elif self.comparison == self.Comparison.NOT_EQUAL:
            return self.a != self.b
        elif self.comparison == self.Comparison.GREATER_THAN:
            return self.a > self.b
        elif self.comparison == self.Comparison.LESS_THAN:
            return self.a < self.b
        elif self.comparison == self.Comparison.GREATER_THAN_OR_EQUAL:
            return self.a >= self.b
        elif self.comparison == self.Comparison.LESS_THAN_OR_EQUAL:
            return self.a <= self.b
        else:
            raise ValueError(f"Unsupported comparison: {self.comparison}")


class IsNone(BaseNode):
    """
    Check if value is None/null.

    Tests whether the input value is Python None. Useful for detecting missing
    or uninitialized values.

    Parameters:
    - value (required): Value to check for None

    Returns: Boolean - true if value is None, false otherwise

    Typical usage: Validate optional parameters, handle missing data, or implement
    null checks before processing. Follow with If node for conditional handling or
    default value nodes.

    null, none, check
    """

    value: Any = Field(default=(), description="The value to check for None")

    async def process(self, context: ProcessingContext) -> bool:
        return self.value is None


class IsIn(BaseNode):
    """
    Check if value exists in list of options (membership test).

    Tests whether value is present in the options list using Python 'in' operator.
    Value equality is used for comparison.

    Parameters:
    - value (required): Value to search for
    - options (required): List of possible values

    Returns: Boolean - true if value found in options, false otherwise

    Typical usage: Validate against allowed values, implement category checks, or
    filter by inclusion criteria. Follow with If node for conditional processing or
    error handling.

    membership, contains, check
    """

    value: Any = Field(default=(), description="The value to check for membership")
    options: list[Any] = Field(
        default=[], description="The list of options to check against"
    )

    async def process(self, context: ProcessingContext) -> bool:
        return self.value in self.options


class All(BaseNode):
    """
    Check if all boolean values in list are true (logical AND across list).

    Returns true only if every boolean in the list is true. Empty list returns true
    (vacuous truth). Equivalent to logical AND of all elements.

    Parameters:
    - values (required): List of boolean values

    Returns: Boolean - true if all values are true, false if any is false

    Typical usage: Ensure all conditions met, validate multiple criteria together,
    or implement comprehensive checks. Precede with multiple comparison nodes.
    Follow with If node for branching.

    boolean, all, check, logic, condition, flow-control, branch
    """

    values: list[bool] = Field(
        default=[], description="List of boolean values to check"
    )

    async def process(self, context: ProcessingContext) -> bool:
        return all(self.values)


class Some(BaseNode):
    """
    Check if any boolean value in list is true (logical OR across list).

    Returns true if at least one boolean in the list is true. Empty list returns
    false. Equivalent to logical OR of all elements.

    Parameters:
    - values (required): List of boolean values

    Returns: Boolean - true if any value is true, false if all are false

    Typical usage: Check if at least one condition met, implement optional criteria,
    or create flexible validation. Precede with multiple comparison nodes. Follow
    with If node for branching.

    boolean, any, check, logic, condition, flow-control, branch
    """

    values: list[bool] = Field(
        default=[], description="List of boolean values to check"
    )

    async def process(self, context: ProcessingContext) -> bool:
        return any(self.values)
