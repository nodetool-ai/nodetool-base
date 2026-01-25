"""
NTL Parser - Parses NTL (NodeTool Language) workflow files.

This module provides a lexer and parser for the NTL syntax, producing
an Abstract Syntax Tree (AST) that can be converted to a workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class TokenType(Enum):
    """Token types for the NTL lexer."""

    # Literals
    STRING = auto()
    NUMBER = auto()
    BOOLEAN = auto()
    IDENTIFIER = auto()

    # Keywords/Symbols
    AT = auto()  # @
    COLON = auto()  # :
    EQUALS = auto()  # =
    DOT = auto()  # .
    ARROW = auto()  # ->
    COMMA = auto()  # ,
    LBRACE = auto()  # {
    RBRACE = auto()  # }
    LBRACKET = auto()  # [
    RBRACKET = auto()  # ]
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
    """Workflow metadata from @key value declarations."""

    name: str | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class NTLProperty:
    """A property assignment within a node definition."""

    name: str
    value: Any
    line: int


@dataclass
class NTLNode:
    """A node definition in the NTL AST."""

    id: str
    type: str
    properties: list[NTLProperty] = field(default_factory=list)
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
class NTLAST:
    """The complete Abstract Syntax Tree for an NTL file."""

    metadata: NTLMetadata = field(default_factory=NTLMetadata)
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

    def read_string(self) -> str:
        """Read a quoted string."""
        quote = self.advance()  # skip opening quote
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
        return "".join(result)

    def read_number(self) -> int | float:
        """Read a number (integer or float)."""
        start = self.pos
        if self.peek() == "-":
            self.advance()
        while self.peek().isdigit():
            self.advance()
        if self.peek() == "." and self.peek(1).isdigit():
            self.advance()
            while self.peek().isdigit():
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
                value = self.read_string()
                self.add_token(TokenType.STRING, value)
            elif char.isdigit() or (char == "-" and self.peek(1).isdigit()):
                value = self.read_number()
                self.add_token(TokenType.NUMBER, value)
            elif char == "@":
                self.advance()
                self.add_token(TokenType.AT)
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
            if self.peek().type == TokenType.AT:
                self.parse_metadata()
            elif self.peek().type == TokenType.IDENTIFIER:
                # Could be node definition or edge definition
                # Look ahead to determine
                if self.is_edge_definition():
                    self.parse_edge()
                else:
                    self.parse_node()
            elif self.peek().type == TokenType.NEWLINE:
                self.advance()
            else:
                raise NTLParseError(
                    f"Unexpected token: {self.peek().type.name}",
                    self.peek().line,
                    self.peek().column,
                )

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

    def parse_metadata(self) -> None:
        """Parse @key value metadata."""
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

        # Parse properties if indented block follows
        if self.peek().type == TokenType.INDENT:
            self.advance()
            node.properties = self.parse_properties()
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

    def parse_properties(self) -> list[NTLProperty]:
        """Parse property assignments in an indented block."""
        properties = []
        while self.peek().type not in (TokenType.DEDENT, TokenType.EOF):
            if self.peek().type == TokenType.NEWLINE:
                self.advance()
                continue
            if self.peek().type == TokenType.IDENTIFIER:
                line = self.peek().line
                name = self.expect(TokenType.IDENTIFIER).value
                self.expect(TokenType.EQUALS)
                value = self.parse_value()
                properties.append(NTLProperty(name=name, value=value, line=line))
                self.skip_newlines()
            else:
                break
        return properties

    def parse_value(self) -> Any:
        """Parse a value (string, number, boolean, reference, object, list)."""
        token = self.peek()

        if token.type == TokenType.STRING:
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
            # Could be an unquoted identifier used as value (e.g., enum values)
            # or comma-separated tags
            return self.parse_identifier_or_tags()
        else:
            raise NTLParseError(
                f"Expected value, got {token.type.name}",
                token.line,
                token.column,
            )

    def parse_identifier_or_tags(self) -> str | list[str]:
        """Parse an identifier or comma-separated list of identifiers."""
        first = self.expect(TokenType.IDENTIFIER).value
        if self.peek().type == TokenType.COMMA:
            # It's a list of identifiers (e.g., for tags)
            items = [first]
            while self.peek().type == TokenType.COMMA:
                self.advance()
                items.append(self.expect(TokenType.IDENTIFIER).value)
            return items
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
            items.append(self.parse_value())
            self.skip_whitespace_tokens()
            if self.peek().type == TokenType.COMMA:
                self.advance()
                self.skip_whitespace_tokens()

        self.expect(TokenType.RBRACKET)
        return items

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
