from pydantic import Field
import typing
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode
import nodetool.nodes.nodetool.mail


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


class SendEmail(GraphNode):
    """Send a plain text email via SMTP.
    email, smtp, send
    """

    smtp_server: str | GraphNode | tuple[GraphNode, str] = Field(
        default="smtp.gmail.com", description="SMTP server hostname"
    )
    smtp_port: int | GraphNode | tuple[GraphNode, str] = Field(
        default=587, description="SMTP server port"
    )
    username: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="SMTP username"
    )
    password: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="SMTP password"
    )
    from_address: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Sender email address"
    )
    to_address: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Recipient email address"
    )
    subject: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Email subject"
    )
    body: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Email body"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.mail.SendEmail"


class EmailFields(GraphNode):
    """
    Decomposes an email into its individual components.
    email, decompose, extract

    Takes an Email object and returns its individual fields:
    - id: Message ID
    - subject: Email subject
    - sender: Sender address
    - date: Datetime of email
    - body: Email body content
    """

    email: types.Email | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Email(
            type="email",
            id="",
            sender="",
            subject="",
            date=types.Datetime(
                type="datetime",
                year=0,
                month=0,
                day=0,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
                tzinfo="UTC",
                utc_offset=0,
            ),
            body="",
        ),
        description="Email object to decompose",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.mail.EmailFields"


class GmailSearch(GraphNode):
    """
    Searches Gmail using Gmail-specific search operators and yields matching emails.
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
