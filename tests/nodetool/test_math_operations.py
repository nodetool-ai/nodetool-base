import pytest
from nodetool.dsl.graph import graph_result
from nodetool.dsl.lib.math import Add, Multiply, Divide, MathFunction, Power
from nodetool.dsl.nodetool.output import FloatOutput

# Example 1: Basic arithmetic (2 + 3) * 4
add_node = Add(a=2, b=3)
multiply_node = Multiply(a=add_node.output, b=4)
basic_arithmetic = FloatOutput(
    name="basic_arithmetic",
    value=multiply_node.output,
)

# Example 2: Trigonometric calculation with power: sin(π/4)²
sine_node = MathFunction(input=3.14159 / 4, operation=MathFunction.Operation.SINE)
square_sine = MathFunction(
    input=sine_node.output,
    operation=MathFunction.Operation.SQUARE,
)
trig_calculation = FloatOutput(
    name="trig_calculation",
    value=square_sine.output,
)

# Example 3: Complex formula: √(a² + b²) - Pythagorean theorem
square_a = MathFunction(input=3, operation=MathFunction.Operation.SQUARE)
square_b = MathFunction(input=4, operation=MathFunction.Operation.SQUARE)
sum_squares = Add(
    a=square_a.output,
    b=square_b.output,
)
sqrt_sum = MathFunction(
    input=sum_squares.output,
    operation=MathFunction.Operation.SQUARE_ROOT,
)
pythagorean = FloatOutput(
    name="pythagorean",
    value=sqrt_sum.output,
)

# Example 4: Nested operations: (10 + 5) / (2 * 3)
numerator = Add(a=10, b=5)
denominator = Multiply(a=2, b=3)
divide_node = Divide(a=numerator.output, b=denominator.output)
nested_operations = FloatOutput(
    name="nested_operations",
    value=divide_node.output,
)

# Example 5: Combining multiple operations: sin(x²) + √(x)
x_squared = MathFunction(input=2, operation=MathFunction.Operation.SQUARE)
sin_x_squared = MathFunction(
    input=x_squared.output,
    operation=MathFunction.Operation.SINE,
)
sqrt_x = MathFunction(input=2, operation=MathFunction.Operation.SQUARE_ROOT)
add_sin_sqrt = Add(
    a=sin_x_squared.output,
    b=sqrt_x.output,
)
combined_operations = FloatOutput(
    name="combined_operations",
    value=add_sin_sqrt.output,
)


@pytest.mark.asyncio
async def test_basic_arithmetic():
    result = await graph_result(basic_arithmetic)
    assert result["basic_arithmetic"] == 20  # (2 + 3) * 4 = 20


@pytest.mark.asyncio
async def test_pythagorean():
    result = await graph_result(pythagorean)
    assert pytest.approx(result["pythagorean"], 0.0001) == 5.0  # √(3² + 4²) = 5


@pytest.mark.asyncio
async def test_nested_operations():
    result = await graph_result(nested_operations)
    assert result["nested_operations"] == 2.5  # (10 + 5) / (2 * 3) = 15 / 6 = 2.5


@pytest.mark.asyncio
async def test_trig_calculation():
    result = await graph_result(trig_calculation)
    assert pytest.approx(result["trig_calculation"], 0.0001) == 0.5  # sin(π/4)² ≈ 0.5


@pytest.mark.asyncio
async def test_combined_operations():
    result = await graph_result(combined_operations)
    assert isinstance(result["combined_operations"], (int, float))
    assert result["combined_operations"] > 0
