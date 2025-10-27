#!/bin/bash
#
# Convenient shell script for running NodeTool evaluations
#
# Usage:
#   ./eval.sh data_generator
#   ./eval.sh data_generator --models openai/gpt-4
#   ./eval.sh data_generator --output results.json
#

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.11+"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON=${PYTHON:-python3}
if ! $PYTHON --version &> /dev/null; then
    PYTHON=python
fi

# Show help if no arguments or help requested
if [[ $# -eq 0 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    cat << EOF
NodeTool Evaluation Runner

Usage: ./eval.sh <eval_type> [options]

Evaluation Types:
  data_generator, data, dg    Evaluate DataGenerator node

Options:
  --models MODELS             Comma-separated list of models (provider/model_id)
  --output FILE               Save results to JSON file
  --help                      Show this help message

Examples:
  ./eval.sh data_generator
  ./eval.sh data --models openai/gpt-4
  ./eval.sh dg --output results.json
  ./eval.sh data_generator --models openai/gpt-4,anthropic/claude-3-5-haiku-20241022 --output comparison.json

EOF
    exit 0
fi

# Get evaluation type
eval_type="$1"
shift

# Call the Python runner
case "$eval_type" in
    data_generator|data|dg)
        $PYTHON "$SCRIPT_DIR/run_eval.py" data_generator "$@"
        ;;
    *)
        echo "❌ Unknown evaluation type: $eval_type"
        echo "Use './eval.sh --help' for usage information"
        exit 1
        ;;
esac
