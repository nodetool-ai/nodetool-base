import argparse
import json
import os
import re
import sys

import openai
from openai.types.shared_params.response_format_json_schema import (
    ResponseFormatJSONSchema,
)


def load_metadata_json():
    """Load node metadata from the package metadata JSON file."""
    metadata_path = os.path.join(
        os.path.dirname(__file__), "src/nodetool/package_metadata/nodetool-base.json"
    )
    with open(metadata_path, "r") as f:
        data = json.load(f)
    return data.get("nodes", [])


JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Workflow Schema",
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "access": {"type": "string"},
        "created_at": {"type": "string", "format": "date-time"},
        "updated_at": {"type": "string", "format": "date-time"},
        "name": {"type": "string"},
        "description": {"type": "string"},
        "tags": {
            "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]
        },
        "thumbnail": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "thumbnail_url": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "graph": {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "parent_id": {
                                "anyOf": [{"type": "string"}, {"type": "null"}]
                            },
                            "type": {"type": "string"},
                            "data": {"type": "object"},
                            "ui_properties": {
                                "type": "object",
                                "properties": {
                                    "position": {
                                        "type": "object",
                                        "properties": {
                                            "x": {"type": "number"},
                                            "y": {"type": "number"},
                                        },
                                        "required": ["x", "y"],
                                    },
                                    "zIndex": {"type": "number"},
                                    "width": {"type": "number"},
                                    "height": {"type": "number"},
                                    "selectable": {"type": "boolean"},
                                },
                                "required": [
                                    "position",
                                    "zIndex",
                                    "width",
                                    "selectable",
                                ],
                            },
                            "dynamic_properties": {"type": "object"},
                        },
                        "required": [
                            "id",
                            "type",
                            "data",
                            "ui_properties",
                            "dynamic_properties",
                        ],
                    },
                },
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "source": {"type": "string"},
                            "sourceHandle": {"type": "string"},
                            "target": {"type": "string"},
                            "targetHandle": {"type": "string"},
                            "ui_properties": {
                                "type": "object",
                                "properties": {"className": {"type": "string"}},
                            },
                        },
                        "required": [
                            "id",
                            "source",
                            "sourceHandle",
                            "target",
                            "targetHandle",
                            "ui_properties",
                        ],
                    },
                },
            },
            "required": ["nodes", "edges"],
        },
        "input_schema": {"anyOf": [{"type": "object"}, {"type": "null"}]},
        "output_schema": {"anyOf": [{"type": "object"}, {"type": "null"}]},
    },
    "required": [
        "id",
        "access",
        "created_at",
        "updated_at",
        "name",
        "description",
        "graph",
    ],
}

MAX_TOOL_ITERATIONS = 8
_NON_ALNUM = re.compile(r"[^a-zA-Z0-9]+")
_WS = re.compile(r"\s+")


def _normalize_text(value: str) -> str:
    value = _NON_ALNUM.sub(" ", value.lower())
    return _WS.sub(" ", value).strip()


def _type_matches(type_meta: dict, type_str: str) -> bool:
    if not type_meta:
        return False
    meta_type = type_meta.get("type")
    if meta_type == "any":
        return True
    if meta_type == "union":
        return any(
            _type_matches(arg, type_str) for arg in type_meta.get("type_args", [])
        )
    if meta_type == "enum":
        return type_str in (type_meta.get("values") or [])
    return meta_type == type_str


def search_nodes_metadata(
    nodes_metadata,
    query,
    include_description=False,
    include_properties=False,
    n_results=10,
    input_type=None,
    output_type=None,
    exclude_namespaces=None,
):
    if isinstance(query, str):
        query = [query]
    if not isinstance(query, list):
        raise ValueError("query must be a list of strings")
    query_tokens = [_normalize_text(item) for item in query]
    query_tokens = [token for token in query_tokens if token]
    exclude_namespaces = exclude_namespaces or []
    scored = []

    for node in nodes_metadata:
        if node.get("namespace") in exclude_namespaces:
            continue

        properties = node.get("properties") or []
        outputs = node.get("outputs") or []
        if input_type:
            if not any(
                _type_matches(prop.get("type", {}), input_type) for prop in properties
            ):
                continue
        if output_type:
            if not any(
                _type_matches(out.get("type", {}), output_type) for out in outputs
            ):
                continue

        title = _normalize_text(node.get("title", ""))
        name = _normalize_text(node.get("node_type", ""))
        desc = _normalize_text(node.get("description", ""))
        text = f"{title} {name} {desc}".strip()
        score = sum(1 for token in query_tokens if token in text)
        if score <= 0:
            continue
        scored.append((score, node))

    scored.sort(key=lambda item: item[0], reverse=True)
    results = [node for _, node in scored[:n_results]]

    if not include_description and not include_properties:
        return [node.get("node_type") for node in results]

    enriched = []
    for node in results:
        item = {"node_type": node.get("node_type")}
        if include_description:
            item["title"] = node.get("title")
            item["description"] = node.get("description")
        if include_properties:
            item["properties"] = [
                {
                    "name": prop.get("name"),
                    "type": (prop.get("type") or {}).get("type"),
                    "description": prop.get("description"),
                    "default": prop.get("default"),
                    "required": bool(prop.get("required", False)),
                }
                for prop in node.get("properties") or []
            ]
            item["outputs"] = [
                {
                    "name": out.get("name"),
                    "type": (out.get("type") or {}).get("type"),
                }
                for out in node.get("outputs") or []
            ]
        enriched.append(item)

    return enriched


def search_examples_data(examples, query, n_results=5, include_content=False):
    if isinstance(query, str):
        query = [query]
    if not isinstance(query, list):
        raise ValueError("query must be a list of strings")
    query_tokens = [_normalize_text(item) for item in query]
    query_tokens = [token for token in query_tokens if token]
    results = []

    for idx, example in enumerate(examples):
        name = _normalize_text(example.get("name", ""))
        desc = _normalize_text(example.get("description", ""))
        text = f"{name} {desc}".strip()
        if not any(token in text for token in query_tokens):
            continue
        item = {
            "index": idx,
            "name": example.get("name"),
            "description": example.get("description"),
        }
        if include_content:
            item["content"] = example
        results.append(item)
        if len(results) >= n_results:
            break

    return results


def _build_tool_definitions():
    return [
        {
            "type": "function",
            "function": {
                "name": "search_nodes",
                "description": (
                    "Search Nodetool nodes by keyword. Returns node_type strings by default. "
                    "Set include_description/include_properties for more detail."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "array", "items": {"type": "string"}},
                        "include_description": {"type": "boolean", "default": False},
                        "include_properties": {"type": "boolean", "default": False},
                        "n_results": {"type": "integer", "default": 10},
                        "input_type": {"type": "string"},
                        "output_type": {"type": "string"},
                        "exclude_namespaces": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_examples",
                "description": (
                    "Search example workflows by keyword and return matching names/descriptions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "array", "items": {"type": "string"}},
                        "n_results": {"type": "integer", "default": 5},
                        "include_content": {"type": "boolean", "default": False},
                    },
                    "required": ["query"],
                },
            },
        },
    ]


def _run_tool_call(tool_name, tool_args, nodes_metadata, examples):
    if tool_name == "search_nodes":
        return search_nodes_metadata(nodes_metadata, **tool_args)
    if tool_name == "search_examples":
        return search_examples_data(examples, **tool_args)
    return {"error": f"Unknown tool: {tool_name}"}


def load_example_jsons(directory):
    """
    Load and parse all JSON files from the given directory.

    Args:
        directory (str): Path to the directory containing JSON example files.

    Returns:
        list: List of parsed JSON objects.
    """
    examples = []
    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        return examples

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, "r") as f:
                    content = json.load(f)
                    examples.append(content)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return examples


def build_context_message(examples, nodes_metadata=None):
    """
    Build a context string from the provided example JSON objects and nodes metadata.

    Args:
        examples (list): List of JSON objects.
        nodes_metadata (dict): Nodes metadata JSON object.

    Returns:
        str: A string that contains formatted examples and metadata.
    """
    context = "JSON Examples:\n\n"
    for example in examples:
        context += json.dumps(example, indent=2)
        context += "\n\n"

    if nodes_metadata:
        context += "Nodes Metadata:\n"
        context += json.dumps(nodes_metadata, indent=2)
        context += "\n\n"

    return context


def generate_workflow(user_prompt, examples):
    """
    Combine the JSON examples with the user prompt and send the request to the OpenAI API.

    Args:
        user_prompt (str): The prompt provided by the user.
        examples (list): List of example JSON objects.

    Returns:
        str: The generated workflow from the API.
    """
    nodes_metadata = load_metadata_json()
    assert isinstance(nodes_metadata, list)
    summarized_nodes = [
        {
            "node_type": node["node_type"],
            "properties": [p["name"] for p in node["properties"]],
        }
        for node in nodes_metadata
    ]
    context_message = build_context_message(examples, summarized_nodes)
    # Combine the JSON context with the user prompt.
    full_prompt = (
        f"{context_message}\n"
        f"User Prompt: {user_prompt}\n\n"
        "Please generate a workflow in JSON format based on the above examples and prompt."
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a NodeTool workflow generator. Build a Directed Acyclic Graph (DAG) "
                "where nodes are operations and edges are typed data flows. Follow NodeTool "
                "cookbook patterns and core concepts:\n"
                "- Use input nodes for user parameters and output nodes for final results.\n"
                "- Keep types compatible across edges, avoid cycles, and connect required inputs.\n"
                "- Prefer streaming patterns for LLM/agent nodes and add Preview nodes when helpful.\n"
                "- Choose a pattern that matches the task:\n"
                "  * Simple pipeline (Input -> Process -> Output)\n"
                "  * Agent-driven generation (Agent/ListGenerator/Summarizer)\n"
                "  * Streaming with previews\n"
                "  * RAG (IndexTextChunks + HybridSearch + FormatText + Agent)\n"
                "  * Database persistence (CreateTable + Insert + Query)\n"
                "  * Email/web integration (GmailSearch/FetchRSSFeed/GetRequest)\n"
                "  * Realtime processing (RealtimeAudioInput + RealtimeAgent)\n"
                "  * Multi-modal conversions (Audio/Text/Image/Video chains)\n"
                "  * Data pipeline (GetRequest + ImportCSV + Filter + ChartGenerator)\n"
                "- Use node types exactly as defined; call search_nodes or search_examples if unsure.\n"
                "Return only JSON that matches the provided schema."
            ),
        },
        {"role": "user", "content": full_prompt},
    ]
    tools = _build_tool_definitions()

    try:
        for _ in range(MAX_TOOL_ITERATIONS):
            response = openai.chat.completions.create(
                model="gpt-5.2",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                response_format=ResponseFormatJSONSchema(
                    type="json_schema",
                    json_schema={
                        "name": "workflow",
                        "schema": JSON_SCHEMA,
                    },
                ),
            )
            message = response.choices[0].message
            # Print response metrics
            print(f"Response: {response.usage}")

            if message.tool_calls:
                messages.append(message.model_dump(exclude_none=True))
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments or "{}")
                    except json.JSONDecodeError as exc:
                        tool_result = {
                            "error": f"Invalid JSON arguments for {tool_name}: {exc}"
                        }
                    else:
                        if not isinstance(tool_args, dict):
                            tool_result = {
                                "error": f"Tool arguments for {tool_name} must be an object."
                            }
                        else:
                            try:
                                tool_result = _run_tool_call(
                                    tool_name, tool_args, nodes_metadata, examples
                                )
                            except Exception as exc:
                                tool_result = {
                                    "error": f"Tool {tool_name} failed: {exc}"
                                }
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_result),
                        }
                    )
                continue

            if message.content:
                return message.content

            return "Error: Model returned no content."

        return "Error: Exceeded maximum tool iterations."
    except Exception as e:
        return f"Error calling OpenAI API: {e}"


def main():
    """
    Main function that parses arguments, loads JSON examples,
    calls the OpenAI API, and prints the generated workflow.
    """
    parser = argparse.ArgumentParser(
        description="Generate a workflow based on a given prompt using example JSON files."
    )
    parser.add_argument(
        "--prompt",
        "-p",
        required=True,
        help="The prompt to use for workflow generation.",
    )
    parser.add_argument(
        "--examples-dir",
        "-e",
        default="examples",
        help="Directory containing the example JSON files.",
    )
    args = parser.parse_args()

    # Ensure that OPENAI_API_KEY is available.
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    examples = load_example_jsons(args.examples_dir)
    if not examples:
        print("No valid JSON examples could be loaded. Exiting.")
        sys.exit(1)

    result = generate_workflow(args.prompt, examples)
    print("Generated Workflow:")
    print(result)


if __name__ == "__main__":
    main()
