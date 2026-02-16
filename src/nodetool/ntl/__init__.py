"""
NTL (NodeTool Language) - A token-efficient workflow syntax for nodetool.

This module provides a parser and loader for .ntl files, which use a
simple, elegant syntax for defining nodetool workflows.

NTL v2.0 Features:
- !ntl version directive
- !meta structured metadata block
- !const constants
- Consistent : syntax for all key-value pairs
- Multi-line strings with triple quotes
- Annotations for UI hints

Example:
    from nodetool.ntl import load_workflow_from_ntl

    # Load from file
    workflow = load_workflow_from_ntl("my_workflow.ntl")

    # Load from v2 string
    workflow = load_workflow_from_ntl('''
    !ntl 2.0
    !meta
      name: "My Workflow"

    node1: nodetool.input.StringInput
      value: "hello"
    ''')
"""

from nodetool.ntl.parser import (
    NTL_VERSION,
    NTLParseError,
    parse_ntl,
)
from nodetool.ntl.loader import load_workflow_from_ntl, ntl_to_json

__all__ = [
    "NTL_VERSION",
    "NTLParseError",
    "load_workflow_from_ntl",
    "ntl_to_json",
    "parse_ntl",
]
