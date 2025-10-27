# Evaluation Framework Guide

Quick reference guide for running and managing generator node evaluations in NodeTool.

## Quick Start

### Run a Quick Evaluation

```bash
# Run DataGenerator evaluation with default Ollama model
python evals/eval_data_generator.py

# Run ListGenerator evaluation
python evals/eval_list_generator.py

# Run SVGGenerator evaluation
python evals/eval_svg_generator.py
```

### Run via Unified CLI

```bash
cd evals

# Using aliases
python run_eval.py dg                    # DataGenerator
python run_eval.py lg                    # ListGenerator
python run_eval.py sog                   # StructuredOutputGenerator
python run_eval.py svg                   # SVGGenerator
```

## Specifying Models

```bash
# Single model
python evals/eval_data_generator.py --models ollama/gemma3:4b

# Multiple models (compare)
python evals/eval_data_generator.py --models ollama/gemma3:1b,ollama/gemma3:4b,openai/gpt-4-mini

# Using CLI
python evals/run_eval.py dg --models anthropic/claude-3-haiku
```

## Saving Results

```bash
# Save to JSON file
python evals/eval_data_generator.py --output results.json

# With models and output
python evals/eval_data_generator.py \
  --models ollama/gemma3:1b,ollama/gemma3:4b \
  --output comparison.json
```

## Available Evaluations

### DataGenerator
- **Aliases:** `data_generator`, `data`, `dg`
- **Tests:** Structured data generation (people, products, numbers, emails, cities)
- **Validation:** Row count and semantic checking

### ListGenerator
- **Aliases:** `list_generator`, `list`, `lg`
- **Tests:** Text list generation (movies, foods, skills, countries, books)
- **Validation:** Item count and content checking

### StructuredOutputGenerator
- **Aliases:** `structured_output_generator`, `structured`, `sog`
- **Tests:** JSON generation (person, product, event, location, organization)
- **Validation:** Schema validation and type checking

### SVGGenerator
- **Aliases:** `svg_generator`, `svg`, `svgg`
- **Tests:** Vector graphics (shape, icon, pattern, chart, diagram)
- **Validation:** SVG structure and element checking

## Common Commands

### Run All Evaluations

```bash
for eval in dg lg sog svg; do
  python evals/run_eval.py $eval --output "${eval}_results.json"
done
```

### Compare Models

```bash
python evals/eval_data_generator.py \
  --models ollama/gemma3:1b,ollama/gemma3:4b,openai/gpt-4-mini \
  --output model_comparison.json
```

### Test with Local Ollama

```bash
# Make sure Ollama is running
ollama serve

# In another terminal:
python evals/eval_data_generator.py --models ollama/llama3.2:3b
```

### Test with OpenAI

```bash
export OPENAI_API_KEY=sk-your-key-here
python evals/eval_data_generator.py --models openai/gpt-4-mini
```

### Test with Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
python evals/eval_data_generator.py --models anthropic/claude-3-haiku
```

## Understanding Results

### Accuracy Score
- **0.0 - 0.7:** Test FAILED (red ❌)
- **0.7 - 1.0:** Test PASSED (green ✓)

### Console Output Example
```
[1/5] Generate 5 people with name and age
  ✓ ollama/gemma3:1b         accuracy=90.00% time=5.23s
  ✓ ollama/gemma3:4b         accuracy=95.00% time=8.12s
```

### JSON Output
Each result includes:
- `task_id`: Unique identifier for the test
- `model`: Model that was evaluated
- `correct`: Boolean pass/fail
- `accuracy_score`: 0.0-1.0 accuracy
- `runtime_seconds`: How long it took
- `error_message`: Any errors (if failed)
- `details`: Test-specific validation info

## Provider Configuration

### Ollama (Local)
```bash
# Start Ollama server
ollama serve

# Pull a model
ollama pull gemma3:4b

# Run evaluation
python evals/eval_data_generator.py --models ollama/gemma3:4b
```

### OpenAI
```bash
export OPENAI_API_KEY=sk-...
python evals/eval_data_generator.py --models openai/gpt-4-mini
```

### Anthropic
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python evals/eval_data_generator.py --models anthropic/claude-3-haiku
```

### Gemini
```bash
export GEMINI_API_KEY=...
python evals/eval_data_generator.py --models gemini/gemini-pro
```

## Help & Documentation

```bash
# View general help
python evals/run_eval.py --help

# View help for specific evaluation
python evals/run_eval.py dg --help

# View help for direct script
python evals/eval_data_generator.py --help
```

## Troubleshooting

### "Model not found" error
```bash
# Check available Ollama models
ollama list

# Pull missing model
ollama pull gemma3:4b
```

### "Connection refused" error
```bash
# Make sure Ollama is running
ollama serve

# Or check that API keys are set
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

### Import errors
```bash
# Make sure you're in the right directory
cd /Users/mg/workspace/nodetool-base

# Verify installation
python -c "from evals.eval_data_generator import DataGeneratorRunner; print('OK')"
```

## Performance Notes

- **Single evaluation, 1 model:** 30-120 seconds
- **Single evaluation, 3 models:** 90-360 seconds
- **All 4 evaluations, 3 models:** 6-24 minutes

Run evaluations in parallel to speed things up:
```bash
python evals/run_eval.py dg --output dg.json &
python evals/run_eval.py lg --output lg.json &
python evals/run_eval.py sog --output sog.json &
python evals/run_eval.py svg --output svg.json &
wait
```

## Creating Custom Evaluations

See `generator_eval_framework.md` for detailed instructions on:
- Creating new evaluation test cases
- Implementing validators
- Building custom runner classes
- Integrating with the CLI

## Files Overview

- **evals/common.py** - Shared framework code
- **evals/eval_data_generator.py** - DataGenerator evaluation
- **evals/eval_list_generator.py** - ListGenerator evaluation
- **evals/eval_structured_output_generator.py** - StructuredOutputGenerator evaluation
- **evals/eval_svg_generator.py** - SVGGenerator evaluation
- **evals/run_eval.py** - Unified CLI wrapper

## Additional Resources

- `GENERATOR_EVAL_FRAMEWORK.md` - Complete architecture guide
- `evals/README.md` - Framework documentation
- `evals/QUICKSTART.md` - Quick start guide
- `evals/OLLAMA_GUIDE.md` - Ollama-specific setup
