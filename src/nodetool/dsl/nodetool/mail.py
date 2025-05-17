from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class AddLabel(GraphNode):
    """
    Adds a label to a Gmail message.
    email, gmail, label
    """

    message_id: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Message ID to label"
    )
    label: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Label to add to the message"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.mail.AddLabel"


class EmailIterator(GraphNode):
    """
    Iterates over a list of email message IDs.
    email, gmail, iterate
    """

    message_ids: list[str] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="List of message IDs to iterate over"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.mail.EmailIterator"


import nodetool.nodes.nodetool.mail
import nodetool.nodes.nodetool.mail


class GmailSearch(GraphNode):
    """
    Searches Gmail using Gmail-specific search operators.
    email, gmail, search

    Use cases:
    - Search for emails based on specific criteria
    - Retrieve emails from a specific sender
    - Filter emails by subject, sender, or date
    """

    DateFilter: typing.ClassVar[type] = (
        nodetool.nodes.nodetool.mail.GmailSearch.DateFilter
    )
    GmailFolder: typing.ClassVar[type] = (
        nodetool.nodes.nodetool.mail.GmailSearch.GmailFolder
    )
    from_address: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Sender's email address to search for"
    )
    to_address: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Recipient's email address to search for"
    )
    subject: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Text to search for in email subject"
    )
    body: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Text to search for in email body"
    )
    date_filter: nodetool.nodes.nodetool.mail.GmailSearch.DateFilter = Field(
        default=nodetool.nodes.nodetool.mail.GmailSearch.DateFilter.SINCE_ONE_DAY,
        description="Date filter to search for",
    )
    keywords: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Custom keywords or labels to search for"
    )
    folder: nodetool.nodes.nodetool.mail.GmailSearch.GmailFolder = Field(
        default=nodetool.nodes.nodetool.mail.GmailSearch.GmailFolder.INBOX,
        description="Email folder to search in",
    )
    text: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="General text to search for anywhere in the email"
    )
    max_results: int | GraphNode | tuple[GraphNode, str] = Field(
        default=50, description="Maximum number of emails to return"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.mail.GmailSearch"


class MoveToArchive(GraphNode):
    """
    Moves specified emails to Gmail archive.
    email, gmail, archive
    """

    message_id: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Message ID to archive"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.mail.MoveToArchive"
