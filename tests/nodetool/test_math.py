import pytest
import numpy as np
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.math import (
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulus,
    Sine,
    Cosine,
    Power,
    Sqrt,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "NodeClass, a, b, expected",
    [
        (Add, 2, 3, 5),
        (Subtract, 5, 3, 2),
        (Multiply, 2, 3, 6),
        (Divide, 6, 3, 2),
        (Modulus, 7, 3, 1),
    ],
)
async def test_basic_math_operations(
    context: ProcessingContext, NodeClass, a, b, expected
):
    node = NodeClass(a=a, b=b)
    result = await node.process(context)
    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "NodeClass, input_value, expected",
    [
        (Sine, 0, 0),
        (Sine, np.pi / 2, 1),
        (Cosine, 0, 1),
        (Cosine, np.pi, -1),
    ],
)
async def test_trigonometric_functions(
    context: ProcessingContext, NodeClass, input_value, expected
):
    node = NodeClass(angle_rad=input_value)
    result = await node.process(context)
    assert result == expected


@pytest.mark.asyncio
async def test_power_function(context: ProcessingContext):
    node = Power(base=2, exponent=3)
    result = await node.process(context)
    assert result == 8


@pytest.mark.asyncio
async def test_sqrt_function(context: ProcessingContext):
    node = Sqrt(x=9)
    result = await node.process(context)
    assert result == 3


# Additional tests for MathFunction node
from nodetool.nodes.lib.math import MathFunction


@pytest.mark.asyncio
async def test_math_function_negate(context: ProcessingContext):
    node = MathFunction(input=5, operation=MathFunction.Operation.NEGATE)
    result = await node.process(context)
    assert result == -5


@pytest.mark.asyncio
async def test_math_function_absolute(context: ProcessingContext):
    node = MathFunction(input=-5, operation=MathFunction.Operation.ABSOLUTE)
    result = await node.process(context)
    assert result == 5


@pytest.mark.asyncio
async def test_math_function_square(context: ProcessingContext):
    node = MathFunction(input=4, operation=MathFunction.Operation.SQUARE)
    result = await node.process(context)
    assert result == 16


@pytest.mark.asyncio
async def test_math_function_cube(context: ProcessingContext):
    node = MathFunction(input=3, operation=MathFunction.Operation.CUBE)
    result = await node.process(context)
    assert result == 27


@pytest.mark.asyncio
async def test_math_function_square_root(context: ProcessingContext):
    node = MathFunction(input=16, operation=MathFunction.Operation.SQUARE_ROOT)
    result = await node.process(context)
    assert result == 4


@pytest.mark.asyncio
async def test_math_function_cube_root(context: ProcessingContext):
    node = MathFunction(input=27, operation=MathFunction.Operation.CUBE_ROOT)
    result = await node.process(context)
    assert abs(result - 3) < 0.0001


@pytest.mark.asyncio
async def test_math_function_trig(context: ProcessingContext):
    # Sine
    node = MathFunction(input=0, operation=MathFunction.Operation.SINE)
    result = await node.process(context)
    assert abs(result) < 0.0001
    
    # Cosine
    node = MathFunction(input=0, operation=MathFunction.Operation.COSINE)
    result = await node.process(context)
    assert abs(result - 1) < 0.0001
    
    # Tangent
    node = MathFunction(input=0, operation=MathFunction.Operation.TANGENT)
    result = await node.process(context)
    assert abs(result) < 0.0001


@pytest.mark.asyncio
async def test_math_function_inverse_trig(context: ProcessingContext):
    # Arcsine
    node = MathFunction(input=0, operation=MathFunction.Operation.ARCSINE)
    result = await node.process(context)
    assert abs(result) < 0.0001
    
    # Arccosine
    node = MathFunction(input=1, operation=MathFunction.Operation.ARCCOSINE)
    result = await node.process(context)
    assert abs(result) < 0.0001
    
    # Arctangent
    node = MathFunction(input=0, operation=MathFunction.Operation.ARCTANGENT)
    result = await node.process(context)
    assert abs(result) < 0.0001


@pytest.mark.asyncio
async def test_math_function_log(context: ProcessingContext):
    # Test natural logarithm with a simple value
    node = MathFunction(input=2.718281828, operation=MathFunction.Operation.LOG)
    result = await node.process(context)
    assert abs(result - 1) < 0.0001


@pytest.mark.asyncio
async def test_divide_by_zero(context: ProcessingContext):
    node = Divide(a=10, b=0)
    with pytest.raises(ZeroDivisionError):
        await node.process(context)


@pytest.mark.asyncio
async def test_negative_operations(context: ProcessingContext):
    # Test negative numbers
    node = Add(a=-5, b=-3)
    result = await node.process(context)
    assert result == -8
    
    node = Multiply(a=-2, b=3)
    result = await node.process(context)
    assert result == -6


@pytest.mark.asyncio
async def test_float_operations(context: ProcessingContext):
    # Test floating point operations
    node = Add(a=1.5, b=2.3)
    result = await node.process(context)
    assert abs(result - 3.8) < 0.0001
    
    node = Divide(a=7, b=2)
    result = await node.process(context)
    assert result == 3.5
