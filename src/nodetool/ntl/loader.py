"""
NTL Loader - Converts NTL AST to nodetool workflow format.

This module provides functions to load NTL files and convert them
to the workflow dictionary format used by nodetool.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from nodetool.ntl.parser import (
    NTLAST,
    NTLConstRef,
    NTLEdge,
    NTLNode,
    NTLReference,
    NTLTypeCast,
    parse_ntl,
)


def generate_uuid() -> str:
    """Generate a UUID for node/edge IDs."""
    return str(uuid.uuid4())


def resolve_constants(value: Any, constants: dict[str, Any]) -> Any:
    """Resolve constant references in a value."""
    if isinstance(value, NTLConstRef):
        if value.name in constants:
            return constants[value.name]
        # Return the name as-is if not found (may be resolved later)
        return value.name
    elif isinstance(value, NTLTypeCast):
        # Apply type cast
        inner = resolve_constants(value.value, constants)
        if value.type_name == "int":
            return int(inner)
        elif value.type_name == "float":
            return float(inner)
        elif value.type_name == "string" or value.type_name == "str":
            return str(inner)
        elif value.type_name == "bool":
            return bool(inner)
        else:
            return inner
    elif isinstance(value, dict):
        return {k: resolve_constants(v, constants) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_constants(v, constants) for v in value]
    else:
        return value


def convert_value(
    value: Any, node_id: str, prop_name: str, constants: dict[str, Any] | None = None
) -> tuple[Any, list[dict]]:
    """
    Convert a parsed value to workflow format.

    Returns:
        Tuple of (converted_value, list_of_edges_to_add)
    """
    edges = []
    constants = constants or {}

    # First resolve constants and type casts
    value = resolve_constants(value, constants)

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
            conv_v, sub_edges = convert_value(v, node_id, prop_name, constants)
            converted[k] = conv_v
            edges.extend(sub_edges)
        return converted, edges
    elif isinstance(value, list):
        # Recursively convert list items
        converted = []
        for item in value:
            conv_item, sub_edges = convert_value(item, node_id, prop_name, constants)
            converted.append(conv_item)
            edges.extend(sub_edges)
        return converted, edges
    else:
        return value, edges


def convert_node(
    node: NTLNode, constants: dict[str, Any] | None = None
) -> tuple[dict, list[dict]]:
    """
    Convert an NTL node to workflow node format.

    Returns:
        Tuple of (node_dict, list_of_edges)
    """
    constants = constants or {}
    all_edges = []
    data = {}

    for prop in node.properties:
        value, edges = convert_value(prop.value, node.id, prop.name, constants)
        if value is not None:
            data[prop.name] = value
        all_edges.extend(edges)

    # Default UI properties
    ui_props = {
        "selected": False,
        "position": {"x": 0, "y": 0},
        "zIndex": 0,
        "width": 280,
        "selectable": True,
    }

    # Apply annotations to UI properties
    for ann in node.annotations:
        if ann.name == "position":
            if isinstance(ann.value, dict):
                ui_props["position"] = ann.value
        elif ann.name == "width":
            ui_props["width"] = ann.value
        elif ann.name == "height":
            ui_props["height"] = ann.value
        elif ann.name == "collapsed":
            ui_props["collapsed"] = ann.value
        elif ann.name == "color":
            ui_props["color"] = ann.value
        elif ann.name == "ui":
            if isinstance(ann.value, dict):
                ui_props.update(ann.value)

    node_dict = {
        "id": node.id,
        "parent_id": None,
        "type": node.type,
        "data": data,
        "ui_properties": ui_props,
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
    Skips nodes that already have positions set (e.g., via annotations).
    """
    x = 50
    y = 50
    spacing_x = 310
    spacing_y = 200
    per_row = 4

    layout_index = 0
    for node in nodes:
        pos = node["ui_properties"].get("position", {})
        # Only auto-layout if position is not already set (x=0, y=0 means unset)
        if pos.get("x", 0) == 0 and pos.get("y", 0) == 0:
            row = layout_index // per_row
            col = layout_index % per_row
            node["ui_properties"]["position"] = {
                "x": x + col * spacing_x,
                "y": y + row * spacing_y,
            }
            layout_index += 1


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
    constants = ast.constants

    # Convert all nodes
    for ntl_node in ast.nodes:
        node_dict, node_edges = convert_node(ntl_node, constants)
        nodes.append(node_dict)
        edges.extend(node_edges)

    # Convert explicit edges
    for ntl_edge in ast.edges:
        edges.append(convert_edge(ntl_edge))

    # Apply layout (only if positions not set by annotations)
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
    # Handle Path objects directly
    if isinstance(source, Path):
        if source.exists():
            content = source.read_text(encoding="utf-8")
        else:
            raise FileNotFoundError(f"NTL file not found: {source}")
    elif isinstance(source, str):
        # First check for NTL syntax indicators (faster and safer than path check)
        if "\n" in source or "@" in source or "!" in source:
            # Contains NTL syntax indicators - treat as source
            content = source
        elif len(source) > 260:
            # Too long to be a file path, treat as source
            content = source
        else:
            # Try to check if it's a file path
            try:
                path = Path(source)
                if path.exists() and path.is_file():
                    content = path.read_text(encoding="utf-8")
                elif source.endswith(".ntl"):
                    # Looks like a file path but doesn't exist
                    raise FileNotFoundError(f"NTL file not found: {source}")
                else:
                    # Treat as source (could be minimal NTL like just metadata)
                    content = source
            except OSError:
                # Path operations failed, treat as source
                content = source
    else:
        content = str(source)

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
    workflow = load_workflow_from_ntl(source)
    return json.dumps(workflow, indent=2)
