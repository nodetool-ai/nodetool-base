#!/usr/bin/env python3
"""
Convenient CLI wrapper for running NodeTool evaluations.

Usage:
    python run_eval.py data_generator
    python run_eval.py data_generator --models openai/gpt-4
    python run_eval.py data_generator --output results.json
"""

import argparse
import sys
import asyncio
from typing import List, Tuple, Optional


async def run_data_generator_eval(
    models: Optional[List[Tuple[str, str]]] = None,
    output_file: Optional[str] = None,
) -> None:
    """Run DataGenerator evaluation."""
    from eval_data_generator import DataGeneratorRunner, run_evaluation

    runner = DataGeneratorRunner()
    await run_evaluation(runner, models, output_file)


async def run_list_generator_eval(
    models: Optional[List[Tuple[str, str]]] = None,
    output_file: Optional[str] = None,
) -> None:
    """Run ListGenerator evaluation."""
    from eval_list_generator import ListGeneratorRunner, run_evaluation

    runner = ListGeneratorRunner()
    await run_evaluation(runner, models, output_file)


async def run_svg_generator_eval(
    models: Optional[List[Tuple[str, str]]] = None,
    output_file: Optional[str] = None,
) -> None:
    """Run SVGGenerator evaluation."""
    from eval_svg_generator import SVGGeneratorRunner, run_evaluation

    runner = SVGGeneratorRunner()
    await run_evaluation(runner, models, output_file)


def main() -> None:
    """Main entry point for evaluation runner."""
    parser = argparse.ArgumentParser(
        description="NodeTool Evaluation Runner - Evaluate multiple generator nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # DataGenerator
  python run_eval.py data_generator
  python run_eval.py dg --models ollama/gemma3:1b

  # ListGenerator
  python run_eval.py list_generator
  python run_eval.py lg --models ollama/gemma3:4b

  # SVGGenerator
  python run_eval.py svg_generator
  python run_eval.py svg --models ollama/llama3.2:3b

Supported evaluations:
  data_generator (aliases: data, dg)              - DataGenerator node
  list_generator (aliases: list, lg)              - ListGenerator node
  svg_generator (aliases: svg, svgg)              - SVGGenerator node
        """,
    )

    # Subcommand for evaluation type
    subparsers = parser.add_subparsers(
        dest="eval_type", help="Type of evaluation to run"
    )

    # Helper function to add common arguments
    def add_common_args(subparser):
        subparser.add_argument(
            "--models",
            type=str,
            default=None,
            help="Comma-separated list of models in format provider/model_id",
        )
        subparser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Output file to save results as JSON",
        )

    # DataGenerator evaluation
    data_gen_parser = subparsers.add_parser(
        "data_generator",
        help="Evaluate DataGenerator node",
        aliases=["data", "dg"],
    )
    add_common_args(data_gen_parser)

    # ListGenerator evaluation
    list_gen_parser = subparsers.add_parser(
        "list_generator",
        help="Evaluate ListGenerator node",
        aliases=["list", "lg"],
    )
    add_common_args(list_gen_parser)

    # SVGGenerator evaluation
    svg_gen_parser = subparsers.add_parser(
        "svg_generator",
        help="Evaluate SVGGenerator node",
        aliases=["svg", "svgg"],
    )
    add_common_args(svg_gen_parser)

    args = parser.parse_args()

    if not args.eval_type:
        parser.print_help()
        sys.exit(1)

    # Parse models if provided
    models: Optional[List[Tuple[str, str]]] = None
    if args.models:
        models = []
        for model_str in args.models.split(","):
            parts = model_str.strip().split("/")
            if len(parts) == 2:
                models.append((parts[0], parts[1]))
            else:
                print(
                    f"⚠️  Invalid model format: {model_str}. Use provider/model_id",
                    file=sys.stderr,
                )
                sys.exit(1)

    # Run the appropriate evaluation
    eval_func = None
    if args.eval_type in ["data_generator", "data", "dg"]:
        eval_func = run_data_generator_eval
    elif args.eval_type in ["list_generator", "list", "lg"]:
        eval_func = run_list_generator_eval
    elif args.eval_type in ["svg_generator", "svg", "svgg"]:
        eval_func = run_svg_generator_eval
    else:
        print(f"Unknown evaluation type: {args.eval_type}", file=sys.stderr)
        sys.exit(1)

    # Run the evaluation asynchronously
    asyncio.run(eval_func(models, args.output))


if __name__ == "__main__":
    main()
