# Control Agent Implementation

This document describes the ControlAgent implementation for nodetool-base, which works with control edges from [nodetool-core PR #587](https://github.com/nodetool-ai/nodetool-core/pull/587).

## Overview

The ControlAgent is a specialized agent node that analyzes context and outputs control parameters that can be used to dynamically configure downstream nodes via control edges.

## Key Concepts

### Control Edges
Control edges are a new edge type (`edge_type="control"`) introduced in nodetool-core that allow parameters to be routed between nodes in a special way:
- Control parameters **override** normal data inputs
- Enable dynamic, context-aware workflow behavior
- Processed by the workflow runner before node execution

### ControlAgent Node
The ControlAgent node (`nodetool.agents.ControlAgent`):
- Receives context as input (text, data, analysis results)
- Uses an LLM to reason about appropriate parameters
- Outputs control parameters via the special `__control_output__` handle
- These parameters can be routed via control edges to downstream nodes

## Implementation Details

### Node Class: `ControlAgent`

Located in: `src/nodetool/nodes/nodetool/agents.py`

**Key Fields:**
- `model`: LanguageModel to use for control decisions
- `system`: System prompt guiding the agent's reasoning
- `context`: The context to analyze (main input)
- `schema_description`: Optional description of expected output parameters
- `max_tokens`: Maximum tokens for LLM response
- `context_window`: Context window size for Ollama models

**Output:**
```python
{
    "__control_output__": {
        "parameter1": value1,
        "parameter2": value2,
        ...
    }
}
```

### Usage Pattern

1. **Create a ControlAgent node** that analyzes context
2. **Connect its `__control_output__`** to downstream nodes via control edges
3. **Control edges** route parameters to the target node's `__control__` handle
4. **Parameters override** normal inputs during node execution

### Example Workflow

```python
from nodetool.dsl.nodetool.agents import ControlAgent
from nodetool.metadata.types import LanguageModel, Provider

# Create a control agent that analyzes image characteristics
control_agent = ControlAgent(
    model=LanguageModel(provider=Provider.Ollama, id="qwen3:4b"),
    context="The image is very dark with low contrast",
    schema_description="brightness: int, contrast: int",
)

# In the workflow graph, create a control edge:
# ControlAgent.__control_output__ --[control edge]--> ImageAdjust.__control__

# The ImageAdjust node receives:
# {"brightness": 30, "contrast": 15}
# These values override its normal data inputs
```

## Testing

### Unit Tests
Located in: `tests/nodetool/test_agents.py`

Tests verify:
- Basic control parameter generation
- JSON parsing and error handling
- Various parameter types (bool, int, float, string)
- Empty context handling

### Integration Tests
Located in: `tests/nodetool/test_control_nodes.py`

Tests verify:
- Control output format for control edges
- Provider integration
- JSON response format request

### Example
Located in: `examples/control_agent_example.py`

Demonstrates:
- Image processing control
- Text generation control
- Workflow routing control

## Architecture Notes

### Control Edge Flow (from nodetool-core PR #587)

1. **Classification**: Runner identifies control edges in `_classify_control_edges()`
2. **Context Building**: Runner builds control context in `_build_control_context()`
3. **Parameter Waiting**: NodeActor waits for control params in `_wait_for_control_params()`
4. **Validation**: Control params validated against node properties
5. **Application**: Control params override normal inputs before processing

### Agent Node Type Convention

Nodes with "Agent" in their type name (e.g., `nodetool.agents.ControlAgent`) are recognized by the workflow runner as special agent nodes that can output control parameters.

## Future Enhancements

Potential improvements:
- Add streaming support for incremental control parameter updates
- Support multiple control outputs for different downstream nodes
- Add control parameter validation against target node schemas
- Add visualization of control edge routing in the UI

## References

- [nodetool-core PR #587](https://github.com/nodetool-ai/nodetool-core/pull/587) - Control edges implementation
- `src/nodetool/nodes/nodetool/agents.py` - ControlAgent implementation
- `tests/nodetool/test_agents.py` - Unit tests
- `examples/control_agent_example.py` - Usage examples
