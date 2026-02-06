"""
NTL Parser - Parses NTL (NodeTool Language) workflow files.

This module provides a lexer and parser for the NTL syntax, producing
an Abstract Syntax Tree (AST) that can be converted to a workflow.

NTL v2 supports:
- Version directive: !ntl 2.0
- Structured metadata: !meta block
- Constants: !const NAME = value
- Consistent syntax: colons for all key-value pairs
- Multi-line strings: triple quotes
- Annotations: !directive for tooling hints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

# NTL format version
NTL_VERSION = "2.0"


class TokenType(Enum):
    """Token types for the NTL lexer."""

    # Literals
    STRING = auto()
    MULTILINE_STRING = auto()
    NUMBER = auto()
    BOOLEAN = auto()
    IDENTIFIER = auto()

    # Keywords/Symbols
    AT = auto()  # @
    BANG = auto()  # !
    COLON = auto()  # :
    EQUALS = auto()  # =
    DOT = auto()  # .
    ARROW = auto()  # ->
    COMMA = auto()  # ,
    LBRACE = auto()  # {
    RBRACE = auto()  # }
    LBRACKET = auto()  # [
    RBRACKET = auto()  # ]
    LPAREN = auto()  # (
    RPAREN = auto()  # )
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    EOF = auto()


@dataclass
class Token:
    """A token produced by the lexer."""

    type: TokenType
    value: Any
    line: int
    column: int


@dataclass
class NTLMetadata:
    """Workflow metadata from !meta block or @key value declarations."""

    name: str | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    version: str | None = None
    author: str | None = None
    schema: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class NTLConstant:
    """A constant definition (!const NAME = value)."""

    name: str
    value: Any
    line: int


@dataclass
class NTLAnnotation:
    """An annotation for a node (!directive: value)."""

    name: str
    value: Any
    line: int


@dataclass
class NTLProperty:
    """A property assignment within a node definition."""

    name: str
    value: Any
    line: int
    annotations: list[NTLAnnotation] = field(default_factory=list)


@dataclass
class NTLNode:
    """A node definition in the NTL AST."""

    id: str
    type: str
    properties: list[NTLProperty] = field(default_factory=list)
    annotations: list[NTLAnnotation] = field(default_factory=list)
    line: int = 0


@dataclass
class NTLEdge:
    """An explicit edge definition (source.handle -> target.handle)."""

    source_node: str
    source_handle: str
    target_node: str
    target_handle: str
    line: int


@dataclass
class NTLReference:
    """A reference to another node's output (@node.handle)."""

    node_id: str
    handle: str


@dataclass
class NTLTypeCast:
    """An explicit type cast (e.g., int(108), float(0.5))."""

    type_name: str
    value: Any


@dataclass
class NTLConstRef:
    """A reference to a constant (e.g., DEFAULT_CUTOFF)."""

    name: str


@dataclass
class NTLAST:
    """The complete Abstract Syntax Tree for an NTL file."""

    version: str | None = None
    metadata: NTLMetadata = field(default_factory=NTLMetadata)
    constants: dict[str, Any] = field(default_factory=dict)
    nodes: list[NTLNode] = field(default_factory=list)
    edges: list[NTLEdge] = field(default_factory=list)


class NTLParseError(Exception):
    """Exception raised when parsing NTL fails."""

    def __init__(self, message: str, line: int, column: int = 0):
        self.line = line
        self.column = column
        super().__init__(f"Line {line}: {message}")


class NTLLexer:
    """Lexer for NTL source code."""

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.indent_stack = [0]
        self.tokens: list[Token] = []
        self.pending_dedents = 0
        self.at_line_start = True

    def peek(self, offset: int = 0) -> str:
        """Peek at character at current position + offset."""
        pos = self.pos + offset
        if pos >= len(self.source):
            return ""
        return self.source[pos]

    def advance(self) -> str:
        """Advance position and return current character."""
        if self.pos >= len(self.source):
            return ""
        char = self.source[self.pos]
        self.pos += 1
        if char == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return char

    def skip_whitespace(self) -> None:
        """Skip spaces and tabs (not newlines)."""
        while self.peek() in " \t":
            self.advance()

    def skip_line_comment(self) -> None:
        """Skip a # comment until end of line."""
        while self.peek() and self.peek() != "\n":
            self.advance()

    def skip_block_comment(self) -> None:
        """Skip a /* */ block comment."""
        self.advance()  # skip /
        self.advance()  # skip *
        while self.pos < len(self.source):
            if self.peek() == "*" and self.peek(1) == "/":
                self.advance()
                self.advance()
                return
            self.advance()
        raise NTLParseError("Unterminated block comment", self.line, self.column)

    def read_string(self) -> tuple[str, bool]:
        """Read a quoted string. Returns (string_value, is_multiline)."""
        quote = self.peek()
        # Check for multi-line string (triple quotes)
        if self.peek(1) == quote and self.peek(2) == quote:
            return self.read_multiline_string(), True

        self.advance()  # skip opening quote
        result = []
        while self.peek() and self.peek() != quote:
            if self.peek() == "\\":
                self.advance()
                escape = self.advance()
                if escape == "n":
                    result.append("\n")
                elif escape == "t":
                    result.append("\t")
                elif escape == "\\":
                    result.append("\\")
                elif escape == '"':
                    result.append('"')
                elif escape == "'":
                    result.append("'")
                else:
                    result.append(escape)
            else:
                result.append(self.advance())
        if not self.peek():
            raise NTLParseError("Unterminated string", self.line, self.column)
        self.advance()  # skip closing quote
        return "".join(result), False

    def read_multiline_string(self) -> str:
        """Read a triple-quoted multi-line string."""
        quote = self.advance()  # first quote
        self.advance()  # second quote
        self.advance()  # third quote

        result = []
        while self.pos < len(self.source):
            if self.peek() == quote and self.peek(1) == quote and self.peek(2) == quote:
                self.advance()
                self.advance()
                self.advance()
                # Strip leading/trailing whitespace from multiline strings
                text = "".join(result)
                # Remove leading newline if present
                if text.startswith("\n"):
                    text = text[1:]
                # Remove trailing whitespace and dedent
                lines = text.rstrip().split("\n")
                if lines:
                    # Find minimum indentation
                    min_indent = len(lines[0]) if lines[0] else 0
                    for line in lines:
                        if line.strip():
                            indent = len(line) - len(line.lstrip())
                            min_indent = min(min_indent, indent)
                    lines = [
                        line[min_indent:] if len(line) >= min_indent else line
                        for line in lines
                    ]
                return "\n".join(lines)
            result.append(self.advance())
        raise NTLParseError("Unterminated multi-line string", self.line, self.column)

    def read_number(self) -> int | float:
        """Read a number (integer or float)."""
        start = self.pos
        if self.peek() == "-":
            self.advance()
        while self.peek() and self.peek().isdigit():
            self.advance()
        peek_char = self.peek()
        peek_next = self.peek(1)
        if peek_char == "." and peek_next and peek_next.isdigit():
            self.advance()
            while self.peek() and self.peek().isdigit():
                self.advance()
            return float(self.source[start : self.pos])
        return int(self.source[start : self.pos])

    def read_identifier(self) -> str:
        """Read an identifier (letters, digits, underscores, dashes)."""
        start = self.pos
        while self.peek() and (self.peek().isalnum() or self.peek() in "_-"):
            self.advance()
        return self.source[start : self.pos]

    def measure_indent(self) -> int:
        """Measure indentation at start of line."""
        indent = 0
        while self.peek() == " ":
            self.advance()
            indent += 1
        while self.peek() == "\t":
            self.advance()
            indent += 4  # treat tab as 4 spaces
        return indent

    def add_token(self, token_type: TokenType, value: Any = None) -> None:
        """Add a token to the list."""
        self.tokens.append(Token(token_type, value, self.line, self.column))

    def tokenize(self) -> list[Token]:
        """Tokenize the entire source."""
        while self.pos < len(self.source):
            if self.at_line_start:
                # Handle indentation
                indent = self.measure_indent()

                # Skip blank lines and comment-only lines
                if self.peek() == "\n":
                    self.advance()
                    continue
                if self.peek() == "#":
                    self.skip_line_comment()
                    if self.peek() == "\n":
                        self.advance()
                    continue
                if self.peek() == "/" and self.peek(1) == "*":
                    self.skip_block_comment()
                    continue

                # Process indent/dedent
                current_indent = self.indent_stack[-1]
                if indent > current_indent:
                    self.indent_stack.append(indent)
                    self.add_token(TokenType.INDENT)
                elif indent < current_indent:
                    while self.indent_stack[-1] > indent:
                        self.indent_stack.pop()
                        self.add_token(TokenType.DEDENT)
                    if self.indent_stack[-1] != indent:
                        raise NTLParseError(
                            "Inconsistent indentation", self.line, self.column
                        )

                self.at_line_start = False

            char = self.peek()

            if char == "\n":
                self.add_token(TokenType.NEWLINE)
                self.advance()
                self.at_line_start = True
            elif char in " \t":
                self.skip_whitespace()
            elif char == "#":
                self.skip_line_comment()
            elif char == "/" and self.peek(1) == "*":
                self.skip_block_comment()
            elif char in "\"'":
                value, is_multiline = self.read_string()
                token_type = (
                    TokenType.MULTILINE_STRING if is_multiline else TokenType.STRING
                )
                self.add_token(token_type, value)
            elif char.isdigit() or (char == "-" and self.peek(1).isdigit()):
                value = self.read_number()
                self.add_token(TokenType.NUMBER, value)
            elif char == "@":
                self.advance()
                self.add_token(TokenType.AT)
            elif char == "!":
                self.advance()
                self.add_token(TokenType.BANG)
            elif char == ":":
                self.advance()
                self.add_token(TokenType.COLON)
            elif char == "=":
                self.advance()
                self.add_token(TokenType.EQUALS)
            elif char == ".":
                self.advance()
                self.add_token(TokenType.DOT)
            elif char == "-" and self.peek(1) == ">":
                self.advance()
                self.advance()
                self.add_token(TokenType.ARROW)
            elif char == ",":
                self.advance()
                self.add_token(TokenType.COMMA)
            elif char == "{":
                self.advance()
                self.add_token(TokenType.LBRACE)
            elif char == "}":
                self.advance()
                self.add_token(TokenType.RBRACE)
            elif char == "[":
                self.advance()
                self.add_token(TokenType.LBRACKET)
            elif char == "]":
                self.advance()
                self.add_token(TokenType.RBRACKET)
            elif char == "(":
                self.advance()
                self.add_token(TokenType.LPAREN)
            elif char == ")":
                self.advance()
                self.add_token(TokenType.RPAREN)
            elif char.isalpha() or char == "_":
                value = self.read_identifier()
                if value in ("true", "false"):
                    self.add_token(TokenType.BOOLEAN, value == "true")
                else:
                    self.add_token(TokenType.IDENTIFIER, value)
            else:
                raise NTLParseError(
                    f"Unexpected character: {char!r}", self.line, self.column
                )

        # Add remaining DEDENTs
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            self.add_token(TokenType.DEDENT)

        self.add_token(TokenType.EOF)
        return self.tokens


class NTLParser:
    """Parser for NTL tokens, producing an AST."""

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0
        self.ast = NTLAST()

    def peek(self, offset: int = 0) -> Token:
        """Peek at token at current position + offset."""
        pos = self.pos + offset
        if pos >= len(self.tokens):
            return self.tokens[-1]  # Return EOF
        return self.tokens[pos]

    def advance(self) -> Token:
        """Advance and return current token."""
        token = self.peek()
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return token

    def expect(self, token_type: TokenType) -> Token:
        """Expect a specific token type."""
        token = self.peek()
        if token.type != token_type:
            raise NTLParseError(
                f"Expected {token_type.name}, got {token.type.name}",
                token.line,
                token.column,
            )
        return self.advance()

    def skip_newlines(self) -> None:
        """Skip any newline tokens."""
        while self.peek().type == TokenType.NEWLINE:
            self.advance()

    def skip_whitespace_tokens(self) -> None:
        """Skip newlines, indents and dedents (used inside objects/lists)."""
        while self.peek().type in (
            TokenType.NEWLINE,
            TokenType.INDENT,
            TokenType.DEDENT,
        ):
            self.advance()

    def parse(self) -> NTLAST:
        """Parse the token stream into an AST."""
        self.skip_newlines()

        while self.peek().type != TokenType.EOF:
            if self.peek().type == TokenType.BANG:
                # Directive: !ntl, !meta, !const, !edges, etc.
                self.parse_directive()
            elif self.peek().type == TokenType.AT:
                # Legacy v1 metadata: @name, @description, @tags
                self.parse_legacy_metadata()
            elif self.peek().type == TokenType.IDENTIFIER:
                # Could be node definition or edge definition
                # Look ahead to determine
                if self.is_edge_definition():
                    self.parse_edge()
                else:
                    self.parse_node()
            elif self.peek().type == TokenType.NEWLINE:
                self.advance()
            elif self.peek().type in (TokenType.INDENT, TokenType.DEDENT):
                # Skip stray indent/dedent at top level
                self.advance()
            else:
                raise NTLParseError(
                    f"Unexpected token: {self.peek().type.name}",
                    self.peek().line,
                    self.peek().column,
                )

        return self.ast

    def parse_directive(self) -> None:
        """Parse a ! directive like !ntl, !meta, !const."""
        self.expect(TokenType.BANG)
        directive = self.expect(TokenType.IDENTIFIER).value

        if directive == "ntl":
            # Version declaration: !ntl 2.0
            version = self.parse_value()
            self.ast.version = str(version)
        elif directive == "meta":
            # Structured metadata block
            self.parse_meta_block()
        elif directive == "const":
            # Constant definition: !const NAME = value
            self.parse_const()
        elif directive == "edges":
            # Explicit edges block (optional)
            self.parse_edges_block()
        else:
            # Unknown directive - store in metadata extra
            if self.peek().type == TokenType.COLON:
                self.advance()
            value = self.parse_value()
            self.ast.metadata.extra[f"!{directive}"] = value

        self.skip_newlines()

    def parse_meta_block(self) -> None:
        """Parse a !meta block with structured metadata."""
        self.skip_newlines()

        if self.peek().type == TokenType.INDENT:
            self.advance()
            while self.peek().type not in (TokenType.DEDENT, TokenType.EOF):
                if self.peek().type == TokenType.NEWLINE:
                    self.advance()
                    continue
                if self.peek().type == TokenType.IDENTIFIER:
                    key = self.expect(TokenType.IDENTIFIER).value
                    self.expect(TokenType.COLON)
                    value = self.parse_value()

                    if key == "name":
                        self.ast.metadata.name = value
                    elif key == "description":
                        self.ast.metadata.description = value
                    elif key == "version":
                        self.ast.metadata.version = str(value)
                    elif key == "author":
                        self.ast.metadata.author = value
                    elif key == "schema":
                        self.ast.metadata.schema = value
                    elif key == "tags":
                        if isinstance(value, list):
                            self.ast.metadata.tags = value
                        elif isinstance(value, str):
                            self.ast.metadata.tags = [
                                t.strip() for t in value.split(",")
                            ]
                        else:
                            self.ast.metadata.tags = [value]
                    else:
                        self.ast.metadata.extra[key] = value

                    self.skip_newlines()
                else:
                    break
            if self.peek().type == TokenType.DEDENT:
                self.advance()

    def parse_const(self) -> None:
        """Parse a constant definition: !const NAME = value."""
        name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.EQUALS)
        value = self.parse_value()
        self.ast.constants[name] = value

    def parse_edges_block(self) -> None:
        """Parse an !edges block with explicit edge definitions."""
        self.skip_newlines()

        if self.peek().type == TokenType.INDENT:
            self.advance()
            while self.peek().type not in (TokenType.DEDENT, TokenType.EOF):
                if self.peek().type == TokenType.NEWLINE:
                    self.advance()
                    continue
                if self.peek().type == TokenType.IDENTIFIER:
                    if self.is_edge_definition():
                        self.parse_edge()
                    else:
                        break
                else:
                    break
            if self.peek().type == TokenType.DEDENT:
                self.advance()

        return self.ast

    def is_edge_definition(self) -> bool:
        """Check if current position is start of edge definition."""
        # Pattern: identifier.identifier -> identifier.identifier
        pos = 0
        if self.peek(pos).type != TokenType.IDENTIFIER:
            return False
        pos += 1
        if self.peek(pos).type != TokenType.DOT:
            return False
        pos += 1
        if self.peek(pos).type != TokenType.IDENTIFIER:
            return False
        pos += 1
        if self.peek(pos).type != TokenType.ARROW:
            return False
        return True

    def parse_legacy_metadata(self) -> None:
        """Parse @key value metadata (v1 legacy syntax)."""
        self.expect(TokenType.AT)
        key = self.expect(TokenType.IDENTIFIER).value
        value = self.parse_value()

        if key == "name":
            self.ast.metadata.name = value
        elif key == "description":
            self.ast.metadata.description = value
        elif key == "tags":
            if isinstance(value, str):
                # Parse comma-separated tags
                self.ast.metadata.tags = [t.strip() for t in value.split(",")]
            elif isinstance(value, list):
                self.ast.metadata.tags = value
            else:
                self.ast.metadata.tags = [value]
        else:
            self.ast.metadata.extra[key] = value

        self.skip_newlines()

    def parse_node(self) -> None:
        """Parse a node definition."""
        line = self.peek().line
        node_id = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLON)

        # Parse node type (dotted identifier)
        node_type = self.parse_dotted_identifier()

        node = NTLNode(id=node_id, type=node_type, line=line)

        self.skip_newlines()

        # Parse properties and annotations if indented block follows
        if self.peek().type == TokenType.INDENT:
            self.advance()
            props, annotations = self.parse_properties_and_annotations()
            node.properties = props
            node.annotations = annotations
            if self.peek().type == TokenType.DEDENT:
                self.advance()

        self.ast.nodes.append(node)
        self.skip_newlines()

    def parse_dotted_identifier(self) -> str:
        """Parse a dotted identifier like nodetool.input.ImageInput."""
        parts = [self.expect(TokenType.IDENTIFIER).value]
        while self.peek().type == TokenType.DOT:
            self.advance()
            parts.append(self.expect(TokenType.IDENTIFIER).value)
        return ".".join(parts)

    def parse_properties_and_annotations(
        self,
    ) -> tuple[list[NTLProperty], list[NTLAnnotation]]:
        """Parse property assignments and annotations in an indented block."""
        properties = []
        annotations = []
        while self.peek().type not in (TokenType.DEDENT, TokenType.EOF):
            if self.peek().type == TokenType.NEWLINE:
                self.advance()
                continue
            if self.peek().type == TokenType.BANG:
                # Annotation: !directive: value
                self.advance()
                ann_name = self.expect(TokenType.IDENTIFIER).value
                self.expect(TokenType.COLON)
                ann_value = self.parse_value()
                annotations.append(
                    NTLAnnotation(name=ann_name, value=ann_value, line=self.peek().line)
                )
                self.skip_newlines()
            elif self.peek().type == TokenType.IDENTIFIER:
                line = self.peek().line
                name = self.expect(TokenType.IDENTIFIER).value
                # Support both = and : for property assignment
                if self.peek().type == TokenType.EQUALS:
                    self.advance()
                elif self.peek().type == TokenType.COLON:
                    self.advance()
                else:
                    raise NTLParseError(
                        f"Expected '=' or ':', got {self.peek().type.name}",
                        self.peek().line,
                        self.peek().column,
                    )
                value = self.parse_value()
                properties.append(NTLProperty(name=name, value=value, line=line))
                self.skip_newlines()
            else:
                break
        return properties, annotations

    def parse_value(self) -> Any:
        """Parse a value (string, number, boolean, reference, object, list, type cast, constant ref)."""
        token = self.peek()

        if token.type == TokenType.STRING:
            self.advance()
            return token.value
        elif token.type == TokenType.MULTILINE_STRING:
            self.advance()
            return token.value
        elif token.type == TokenType.NUMBER:
            self.advance()
            return token.value
        elif token.type == TokenType.BOOLEAN:
            self.advance()
            return token.value
        elif token.type == TokenType.AT:
            return self.parse_reference()
        elif token.type == TokenType.LBRACE:
            return self.parse_object()
        elif token.type == TokenType.LBRACKET:
            return self.parse_list()
        elif token.type == TokenType.IDENTIFIER:
            # Could be:
            # - Type cast: int(42), float(0.5), string("hello")
            # - Constant reference: MY_CONSTANT
            # - Unquoted identifier/enum value
            # - Comma-separated tags
            return self.parse_identifier_or_special()
        else:
            raise NTLParseError(
                f"Expected value, got {token.type.name}",
                token.line,
                token.column,
            )

    def parse_identifier_or_special(self) -> Any:
        """Parse an identifier, type cast, constant ref, or comma-separated list."""
        first = self.expect(TokenType.IDENTIFIER).value

        # Check for type cast: int(value), float(value), string(value)
        if self.peek().type == TokenType.LPAREN:
            self.advance()  # skip (
            inner_value = self.parse_value()
            self.expect(TokenType.RPAREN)
            return NTLTypeCast(type_name=first, value=inner_value)

        # Check for comma-separated list (e.g., for tags)
        if self.peek().type == TokenType.COMMA:
            items = [first]
            while self.peek().type == TokenType.COMMA:
                self.advance()
                items.append(self.expect(TokenType.IDENTIFIER).value)
            return items

        # Check if it's an uppercase constant reference
        if first.isupper():
            # Could be a constant - check if defined
            if first in self.ast.constants:
                return self.ast.constants[first]
            # Return as constant reference for later resolution
            return NTLConstRef(name=first)

        return first

    def parse_reference(self) -> NTLReference:
        """Parse a reference like @node.handle."""
        self.expect(TokenType.AT)
        node_id = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.DOT)
        handle = self.expect(TokenType.IDENTIFIER).value
        return NTLReference(node_id=node_id, handle=handle)

    def parse_object(self) -> dict[str, Any]:
        """Parse an object like {key: value, ...}."""
        self.expect(TokenType.LBRACE)
        self.skip_whitespace_tokens()
        obj = {}

        while self.peek().type != TokenType.RBRACE:
            if self.peek().type in (
                TokenType.NEWLINE,
                TokenType.INDENT,
                TokenType.DEDENT,
            ):
                self.advance()
                continue
            key = self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.COLON)
            value = self.parse_value()
            obj[key] = value
            self.skip_whitespace_tokens()
            if self.peek().type == TokenType.COMMA:
                self.advance()
                self.skip_whitespace_tokens()

        self.expect(TokenType.RBRACE)
        return obj

    def parse_list(self) -> list[Any]:
        """Parse a list like [value, ...]."""
        self.expect(TokenType.LBRACKET)
        self.skip_whitespace_tokens()
        items = []

        while self.peek().type != TokenType.RBRACKET:
            if self.peek().type in (
                TokenType.NEWLINE,
                TokenType.INDENT,
                TokenType.DEDENT,
            ):
                self.advance()
                continue
            # Parse single value (don't consume comma-separated identifiers as list)
            items.append(self.parse_single_value())
            self.skip_whitespace_tokens()
            if self.peek().type == TokenType.COMMA:
                self.advance()
                self.skip_whitespace_tokens()

        self.expect(TokenType.RBRACKET)
        return items

    def parse_single_value(self) -> Any:
        """Parse a single value without consuming comma-separated lists."""
        token = self.peek()

        if token.type == TokenType.STRING:
            self.advance()
            return token.value
        elif token.type == TokenType.MULTILINE_STRING:
            self.advance()
            return token.value
        elif token.type == TokenType.NUMBER:
            self.advance()
            return token.value
        elif token.type == TokenType.BOOLEAN:
            self.advance()
            return token.value
        elif token.type == TokenType.AT:
            return self.parse_reference()
        elif token.type == TokenType.LBRACE:
            return self.parse_object()
        elif token.type == TokenType.LBRACKET:
            return self.parse_list()
        elif token.type == TokenType.IDENTIFIER:
            # Parse identifier but don't consume comma-separated lists
            first = self.expect(TokenType.IDENTIFIER).value
            # Check for type cast
            if self.peek().type == TokenType.LPAREN:
                self.advance()
                inner_value = self.parse_value()
                self.expect(TokenType.RPAREN)
                return NTLTypeCast(type_name=first, value=inner_value)
            # Check if it's an uppercase constant reference
            if first.isupper():
                if first in self.ast.constants:
                    return self.ast.constants[first]
                return NTLConstRef(name=first)
            return first
        else:
            raise NTLParseError(
                f"Expected value, got {token.type.name}",
                token.line,
                token.column,
            )

    def parse_edge(self) -> None:
        """Parse an edge definition like source.handle -> target.handle."""
        line = self.peek().line
        source_node = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.DOT)
        source_handle = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.ARROW)
        target_node = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.DOT)
        target_handle = self.expect(TokenType.IDENTIFIER).value

        edge = NTLEdge(
            source_node=source_node,
            source_handle=source_handle,
            target_node=target_node,
            target_handle=target_handle,
            line=line,
        )
        self.ast.edges.append(edge)
        self.skip_newlines()


def parse_ntl(source: str) -> NTLAST:
    """
    Parse NTL source code into an Abstract Syntax Tree.

    Args:
        source: NTL source code as a string.

    Returns:
        NTLAST: The parsed Abstract Syntax Tree.

    Raises:
        NTLParseError: If parsing fails.
    """
    lexer = NTLLexer(source)
    tokens = lexer.tokenize()
    parser = NTLParser(tokens)
    return parser.parse()
