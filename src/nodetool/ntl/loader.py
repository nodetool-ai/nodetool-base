"""
NTL Loader - Converts NTL AST to nodetool workflow format.

This module provides functions to load NTL files and convert them
to the workflow dictionary format used by nodetool.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from nodetool.ntl.parser import (
    NTLAST,
    NTLEdge,
    NTLNode,
    NTLReference,
    parse_ntl,
)


def generate_uuid() -> str:
    """Generate a UUID for node/edge IDs."""
    return str(uuid.uuid4())


def convert_value(value: Any, node_id: str, prop_name: str) -> tuple[Any, list[dict]]:
    """
    Convert a parsed value to workflow format.

    Returns:
        Tuple of (converted_value, list_of_edges_to_add)
    """
    edges = []

    if isinstance(value, NTLReference):
        # This creates an implicit edge
        edge = {
            "id": generate_uuid(),
            "source": value.node_id,
            "sourceHandle": value.handle,
            "target": node_id,
            "targetHandle": prop_name,
            "ui_properties": None,
        }
        edges.append(edge)
        # Return None as the value since it's connected via edge
        return None, edges
    elif isinstance(value, dict):
        # Recursively convert dict values
        converted = {}
        for k, v in value.items():
            conv_v, sub_edges = convert_value(v, node_id, prop_name)
            converted[k] = conv_v
            edges.extend(sub_edges)
        return converted, edges
    elif isinstance(value, list):
        # Recursively convert list items
        converted = []
        for item in value:
            conv_item, sub_edges = convert_value(item, node_id, prop_name)
            converted.append(conv_item)
            edges.extend(sub_edges)
        return converted, edges
    else:
        return value, edges


def convert_node(node: NTLNode) -> tuple[dict, list[dict]]:
    """
    Convert an NTL node to workflow node format.

    Returns:
        Tuple of (node_dict, list_of_edges)
    """
    all_edges = []
    data = {}

    for prop in node.properties:
        value, edges = convert_value(prop.value, node.id, prop.name)
        if value is not None:
            data[prop.name] = value
        all_edges.extend(edges)

    node_dict = {
        "id": node.id,
        "parent_id": None,
        "type": node.type,
        "data": data,
        "ui_properties": {
            "selected": False,
            "position": {"x": 0, "y": 0},
            "zIndex": 0,
            "width": 280,
            "selectable": True,
        },
        "dynamic_properties": {},
        "dynamic_outputs": {},
        "sync_mode": "on_any",
    }

    return node_dict, all_edges


def convert_edge(edge: NTLEdge) -> dict:
    """Convert an NTL edge to workflow edge format."""
    return {
        "id": generate_uuid(),
        "source": edge.source_node,
        "sourceHandle": edge.source_handle,
        "target": edge.target_node,
        "targetHandle": edge.target_handle,
        "ui_properties": None,
    }


def layout_nodes(nodes: list[dict]) -> None:
    """
    Apply automatic layout to nodes.

    Positions nodes in a simple left-to-right flow based on order.
    """
    x = 50
    y = 50
    spacing_x = 310
    spacing_y = 200
    per_row = 4

    for i, node in enumerate(nodes):
        row = i // per_row
        col = i % per_row
        node["ui_properties"]["position"] = {
            "x": x + col * spacing_x,
            "y": y + row * spacing_y,
        }


def ast_to_workflow(ast: NTLAST) -> dict:
    """
    Convert an NTL AST to a workflow dictionary.

    Args:
        ast: The parsed NTL Abstract Syntax Tree.

    Returns:
        A workflow dictionary compatible with nodetool.
    """
    nodes = []
    edges = []

    # Convert all nodes
    for ntl_node in ast.nodes:
        node_dict, node_edges = convert_node(ntl_node)
        nodes.append(node_dict)
        edges.extend(node_edges)

    # Convert explicit edges
    for ntl_edge in ast.edges:
        edges.append(convert_edge(ntl_edge))

    # Apply layout
    layout_nodes(nodes)

    # Build workflow
    workflow = {
        "id": generate_uuid().replace("-", ""),
        "access": "private",
        "created_at": None,
        "updated_at": None,
        "name": ast.metadata.name or "Untitled Workflow",
        "tool_name": None,
        "description": ast.metadata.description or "",
        "tags": ast.metadata.tags,
        "thumbnail": None,
        "thumbnail_url": None,
        "graph": {
            "nodes": nodes,
            "edges": edges,
        },
        "input_schema": None,
        "output_schema": None,
        "settings": None,
        "package_name": None,
        "path": None,
        "run_mode": None,
        "required_providers": None,
        "required_models": None,
    }

    # Add any extra metadata
    for key, value in ast.metadata.extra.items():
        if key not in workflow:
            workflow[key] = value

    return workflow


def load_workflow_from_ntl(source: str | Path) -> dict:
    """
    Load a workflow from an NTL source.

    Args:
        source: Either an NTL source string or a Path to an .ntl file.

    Returns:
        A workflow dictionary compatible with nodetool.

    Raises:
        NTLParseError: If parsing fails.
        FileNotFoundError: If the file doesn't exist.
    """
    if isinstance(source, Path) or (
        isinstance(source, str) and (source.endswith(".ntl") or "\n" not in source)
    ):
        # Treat as file path
        path = Path(source)
        if path.exists():
            content = path.read_text(encoding="utf-8")
        elif "\n" not in str(source) and not Path(source).suffix:
            # It's a short string without newlines but not a file - could be source
            content = str(source)
        else:
            raise FileNotFoundError(f"NTL file not found: {source}")
    else:
        # Treat as source string
        content = source

    ast = parse_ntl(content)
    return ast_to_workflow(ast)


def ntl_to_json(source: str | Path) -> str:
    """
    Convert NTL source to JSON workflow format.

    Args:
        source: Either an NTL source string or a Path to an .ntl file.

    Returns:
        JSON string of the workflow.
    """
    import json

    workflow = load_workflow_from_ntl(source)
    return json.dumps(workflow, indent=2)
