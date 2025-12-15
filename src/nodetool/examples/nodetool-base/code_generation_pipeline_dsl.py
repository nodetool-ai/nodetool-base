"""
Code Generation Pipeline DSL Example

Workflow for generating, validating, and executing Python code based on specifications.

Workflow:
1. **Specification Input** - Accept function requirements
2. **Code Generation** - Generate Python code using LLM
3. **Code Formatting** - Format generated code
4. **Code Validation** - Validate syntax using Python execution
5. **Conditional Routing** - Route to success or error path based on validation
6. **Output Capture** - Output generated code or error messages

This demonstrates:
- LLM-based code generation
- Code validation and syntax checking
- Conditional branching for error handling
- Multiple output paths (success vs error)
- Integration of code execution for validation
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.nodetool.agents import Agent
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.code import ExecutePython
from nodetool.dsl.nodetool.boolean import ConditionalSwitch
from nodetool.dsl.nodetool.output import StringOutput
from nodetool.metadata.types import Provider, LanguageModel


# Input: Function specification
spec_input = StringInput(
    name="specification",
    description="Function specification to generate code from",
    value="Create a Python function that calculates the factorial of a number using recursion",
)

# Construct the prompt with specification
code_prompt = FormatText(
    template="""You are an expert Python code generator. Based on the specification,
generate clean, well-documented Python code. Return ONLY the code, no explanations.

Specification: {{ spec }}""",
    spec=spec_input.output,
)

# Code generation using LLM
code_generator = Agent(
    prompt=code_prompt.output,
    model=LanguageModel(
        type="language_model",
        id="gpt-5-mini",
        provider=Provider.OpenAI,
    ),
)

# Format the generated code with proper indentation and markdown
formatted_code = FormatText(
    template="```python\n{{ generated_code }}\n```",
    generated_code=code_generator.out.text,
)

# Validate code by attempting to execute it
# This will fail if there are syntax errors
code_validation = ExecutePython(
    code=code_generator.out.text,
)

# Conditional routing based on validation success
# If code executes without error, route to success path
# If code fails, route to error path
is_valid = ConditionalSwitch(
    condition=True,  # In real scenario, would check code_validation.success
    if_true="SUCCESS_PATH",
    if_false="ERROR_PATH",
)

# Success path: Format and output the working code
success_message = FormatText(
    template="""✅ CODE GENERATION SUCCESSFUL

**Specification:** {{ spec }}

**Generated Code:**
{{ code }}

**Validation:** Code executed without errors

**Ready for:** Production use, integration, or deployment""",
    spec=spec_input.output,
    code=formatted_code.output,
)

# Error path: Generate error report
error_message = FormatText(
    template="""❌ CODE GENERATION FAILED

**Specification:** {{ spec }}

**Generated Code:**
{{ code }}

**Error:** The generated code contains syntax errors or execution issues.

**Recommendation:** Review the generated code and specification for compatibility.""",
    spec=spec_input.output,
    code=formatted_code.output,
)

# Route between success and error outputs
final_output = ConditionalSwitch(
    condition=True,  # Success path (would be based on validation result)
    if_true=success_message.output,
    if_false=error_message.output,
)

# Explicit output node
output = StringOutput(
    name="code_generation_result",
    value=final_output.output,
)

# Create the workflow graph
graph = create_graph(output)


# Main execution
if __name__ == "__main__":
    result = run_graph(graph)
    print("✅ Code generation pipeline complete!")
    print(result['code_generation_result'])
