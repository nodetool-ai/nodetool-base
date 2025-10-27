# NodeTool Evaluation Scripts

This directory contains evaluation scripts for testing NodeTool components across multiple language models. Each script tests a specific node or agent type with a curated dataset and calculates accuracy metrics.

## Available Evaluations

### 1. DataGenerator Evaluation (`eval_data_generator.py`)

Tests the `DataGenerator` node, which generates structured data (dataframes) from natural language prompts.

**What it tests:**
- Generating people with names and ages
- Generating products with names, prices, and categories
- Generating random numbers in specific ranges
- Generating valid email addresses
- Generating cities and their countries

**Quick Start:**

```bash
# Run with default local Ollama models
python eval_data_generator.py

# Run with specific models
python eval_data_generator.py --models ollama/gemma3:1b
python eval_data_generator.py --models openai/gpt-4,anthropic/claude-3-5-haiku-20241022

# Save results to JSON
python eval_data_generator.py --output results.json

# Combine options
python eval_data_generator.py \
  --models ollama/gemma3:4b,openai/gpt-4-mini \
  --output data_generator_results.json
```

**Default Models:**
Uses local Ollama models by default:
- `ollama/gemma3:1b` (fast, lightweight - 815 MB)
- `ollama/gemma3:270m` (medium - 291 MB)
- `ollama/gemma3:4b` (larger - 3.3 GB)

**Output:**

The script provides:
- **Per-task accuracy scores** for each model
- **Average accuracy** across all tasks
- **Per-model summaries** showing:
  - Average accuracy percentage
  - Number of correct tests
  - Total tests run
- **Detailed results** saved to JSON (if `--output` is specified)

Example output:
```
🚀 Starting DataGenerator Evaluation
📊 Evaluating 5 tasks
🤖 Models: 2
   openai/gpt-4, openai/gpt-4-mini

[1/5] Generate 5 people with name and age
  ✓ openai/gpt-4              accuracy=95.00% rows=5/5 time=2.45s
  ✓ openai/gpt-4-mini         accuracy=92.00% rows=5/5 time=1.89s

[2/5] Generate 3 products with name, price, and category
  ✓ openai/gpt-4              accuracy=98.00% rows=3/3 time=2.12s
  ✓ openai/gpt-4-mini         accuracy=96.00% rows=3/3 time=1.76s

...

======================================================================
📈 SUMMARY
======================================================================

openai/gpt-4:
  Average Accuracy: 95.60%
  Correct Tests: 5/5

openai/gpt-4-mini:
  Average Accuracy: 93.20%
  Correct Tests: 5/5

🏆 Best Model: openai/gpt-4 (95.60%)
======================================================================
```

**Model Format:**

Models should be specified as `provider/model_id`. Supported providers:
- `ollama` - Local Ollama models (gemma3:1b, gemma3:270m, gemma3:4b, etc.)
- `openai` - OpenAI models (gpt-4, gpt-4-mini, etc.)
- `anthropic` - Anthropic models (claude-3-5-haiku-20241022, etc.)
- `gemini` - Google Gemini models (gemini-2.5-flash, etc.)

### 2. Math Agent Evaluation (`eval_math_agent.py`)

Tests the `SimpleAgent` with math tools for solving deterministic math problems.

**Example:**
```bash
python evals/eval_runner.py --agent math --full
```

### 3. Data Agent Evaluation (`eval_data_agent.py`)

Tests the `SimpleAgent` with data manipulation tools using the Iris dataset.

**Example:**
```bash
python evals/eval_runner.py --agent data --full
```

### 4. Browser Agent Evaluation (`eval_browser_agent.py`)

Tests the `SimpleAgent` with browser tools for web navigation and information extraction.

**Example:**
```bash
python evals/eval_runner.py --agent browser --full
```

### 5. Search Agent Evaluation (`eval_search_agent.py`)

Tests the `SimpleAgent` with Google search tools for information retrieval.

**Example:**
```bash
python evals/eval_runner.py --agent search --full
```

## Unified Evaluation Runner

For agent-based evaluations, use the unified `eval_runner.py`:

```bash
# Run full evaluation for a specific agent
python eval_runner.py --agent math --full
python eval_runner.py --agent data --full

# Run single test
python eval_runner.py --agent math \
  --provider openai \
  --model gpt-4 \
  --problem-json '{"problem":"What is sqrt(2)?"}'
```

## Results Format

### JSON Output Format

When using `--output` flag, results are saved as JSON:

```json
{
  "summary": {
    "openai/gpt-4": {
      "average_accuracy": 0.956,
      "correct_count": 5,
      "total_tests": 5,
      "accuracy_scores": [0.95, 0.98, 0.92, 0.94, 0.96]
    }
  },
  "detailed_results": [
    {
      "task_id": "people_5",
      "model": "openai/gpt-4",
      "correct": true,
      "accuracy_score": 0.95,
      "runtime_seconds": 2.45,
      "error_message": null,
      "row_count": 5
    }
  ]
}
```

## Evaluation Metrics

### DataGenerator Accuracy Calculation

For each task, accuracy is calculated as:

```
accuracy = (row_accuracy + validator_accuracy) / 2
```

Where:
- **row_accuracy**: 1.0 if correct number of rows generated, 0.5 otherwise
- **validator_accuracy**: Average of all validator functions (0 or 1 each)

A result is considered **correct** if accuracy > 70%.

### Validators

Each DataGenerator task uses validators that check:
- **Structure**: Correct number of columns
- **Data types**: Values match expected types (string, int, float)
- **Reasonableness**: Values are within expected ranges
  - Ages: 0-150
  - Prices: 0-10000
  - Numbers: 0-100
  - Emails: Valid email format

## Configuration

### Environment Variables

Control concurrency for parallel evaluations:

```bash
export DATA_GENERATOR_CONCURRENCY=2
export MATH_AGENT_CONCURRENCY=3
```

### API Keys

Ensure API keys are set for the providers you want to use:

```bash
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key
export GOOGLE_API_KEY=your_key
```

## Adding New Evaluations

To add a new evaluation:

1. Create `eval_your_node.py` in this directory
2. Implement:
   - Test case generator function (`generate_eval_cases()`)
   - Validator functions for each task type
   - `evaluate_case()` function to run a single test
   - `run_evaluation()` async function to orchestrate all tests
   - `main()` CLI entry point

3. Add to imports in `eval_runner.py` (optional)

Example structure:

```python
from typing import List, Tuple

@dataclass
class EvalCase:
    task_id: str
    prompt: str
    expected_output: Any
    validators: List[str]

def generate_eval_cases() -> List[EvalCase]:
    return [...]

def validate_example(result: Any) -> Tuple[bool, str]:
    """Return (is_valid, error_message)"""
    return True, "Valid"

async def evaluate_case(case, model, context) -> EvalResult:
    # Run your node and validate
    return EvalResult(...)

async def run_evaluation(models=None, output_file=None):
    # Orchestrate evaluation
    return model_accuracies

def main():
    # CLI interface
    asyncio.run(run_evaluation(...))

if __name__ == "__main__":
    main()
```

## Tips for Better Evaluations

1. **Use deterministic test cases**: Make expected outputs unambiguous
2. **Include edge cases**: Test boundary conditions
3. **Validate structure first**: Check format before semantic validation
4. **Set reasonable timeouts**: Some models may be slow
5. **Log failures**: Include error messages for debugging
6. **Test with multiple models**: Compare across providers and sizes

## Troubleshooting

### Models not found
Ensure you're using correct model IDs and have API access:
```bash
python eval_data_generator.py --models openai/gpt-4
```

### API errors
Check environment variables are set:
```bash
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

### Slow evaluations
Reduce number of tasks or models:
```bash
python eval_data_generator.py --models openai/gpt-4-mini
```

### Out of memory
Run with fewer concurrent tasks or in smaller batches.

## Performance Baseline

Typical runtime for DataGenerator evaluation:

| Models | Tasks | Expected Runtime |
|--------|-------|------------------|
| 1      | 5     | 15-30 seconds    |
| 2      | 5     | 30-60 seconds    |
| 3      | 5     | 45-90 seconds    |

Times vary based on:
- Model latency
- Network conditions
- System resources
- Task complexity

## References

- DataGenerator Node: `src/nodetool/nodes/nodetool/generators.py`
- SimpleAgent: `src/nodetool/agents/simple_agent.py`
- Processing Context: `src/nodetool/workflows/processing_context.py`
