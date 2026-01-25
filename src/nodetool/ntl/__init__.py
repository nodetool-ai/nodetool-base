"""
NTL (NodeTool Language) - A token-efficient workflow syntax for nodetool.

This module provides a parser and loader for .ntl files, which use a
simple, elegant syntax for defining nodetool workflows.

Example:
    from nodetool.ntl import load_workflow_from_ntl

    workflow = load_workflow_from_ntl("my_workflow.ntl")
"""

from nodetool.ntl.parser import parse_ntl, NTLParseError
from nodetool.ntl.loader import load_workflow_from_ntl

__all__ = ["parse_ntl", "load_workflow_from_ntl", "NTLParseError"]
