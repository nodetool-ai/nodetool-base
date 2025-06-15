import asyncio
from datetime import datetime, timedelta
from enum import Enum
from pydantic import Field
from nodetool.common.imap import search_emails, fetch_email
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import (
    DateCriteria,
    DateSearchCondition,
    Datetime,
    Email,
    EmailSearchCriteria,
)


KEYWORD_SEPARATOR_REGEX = r"\s+|,|;"


class EmailFields(BaseNode):
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

    email: Email = Field(default=Email(), description="Email object to decompose")

    @classmethod
    def return_type(cls):
        return {
            "id": str,
            "subject": str,
            "sender": str,
            "date": Datetime,
            "body": str,
        }

    async def process(self, context: ProcessingContext):
        if not self.email:
            raise ValueError("Email is required")

        return {
            "id": self.email.id,
            "subject": self.email.subject,
            "sender": self.email.sender,
            "date": self.email.date,
            "body": self.email.body,
        }


class GmailSearch(BaseNode):
    """
    Searches Gmail using Gmail-specific search operators and yields matching emails.
    email, gmail, search

    Use cases:
    - Search for emails based on specific criteria
    - Retrieve emails from a specific sender
    - Filter emails by subject, sender, or date
    """

    class DateFilter(Enum):
        SINCE_ONE_HOUR = "SINCE_ONE_HOUR"
        SINCE_ONE_DAY = "SINCE_ONE_DAY"
        SINCE_ONE_WEEK = "SINCE_ONE_WEEK"
        SINCE_ONE_MONTH = "SINCE_ONE_MONTH"
        SINCE_ONE_YEAR = "SINCE_ONE_YEAR"

    class GmailFolder(Enum):
        INBOX = "INBOX"
        SENT_MAIL = "[Gmail]/Sent Mail"
        DRAFTS = "[Gmail]/Drafts"
        SPAM = "[Gmail]/Spam"
        TRASH = "[Gmail]/Trash"

    from_address: str = Field(
        default="",
        description="Sender's email address to search for",
    )
    to_address: str = Field(
        default="",
        description="Recipient's email address to search for",
    )
    subject: str = Field(
        default="",
        description="Text to search for in email subject",
    )
    body: str = Field(
        default="",
        description="Text to search for in email body",
    )
    date_filter: DateFilter = Field(
        default=DateFilter.SINCE_ONE_DAY,
        description="Date filter to search for",
    )
    keywords: str = Field(
        default="",
        description="Custom keywords or labels to search for",
    )
    folder: GmailFolder = Field(
        default=GmailFolder.INBOX,
        description="Email folder to search in",
    )
    text: str = Field(
        default="",
        description="General text to search for anywhere in the email",
    )
    max_results: int = Field(
        default=50,
        description="Maximum number of emails to return",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["from_address", "subject", "body", "date_filter", "max_results"]

    @classmethod
    def return_type(cls):
        return {
            "email": Email,
            "message_id": str,
        }

    async def gen_process(self, context: ProcessingContext):
        search_criteria = EmailSearchCriteria(
            from_address=(
                self.from_address.strip() if self.from_address.strip() else None
            ),
            to_address=self.to_address.strip() if self.to_address.strip() else None,
            subject=self.subject.strip() if self.subject.strip() else None,
            body=self.body.strip() if self.body.strip() else None,
            date_condition=get_date_condition(self.date_filter),
            keywords=(
                [
                    k.strip()
                    for k in self.keywords.split(KEYWORD_SEPARATOR_REGEX)
                    if k.strip()
                ]
                if self.keywords and self.keywords.strip()
                else []
            ),
            folder=self.folder.value if self.folder else None,
            text=self.text.strip() if self.text and self.text.strip() else None,
        )

        message_ids = search_emails(
            context.get_gmail_connection(), search_criteria, self.max_results
        )

        for message_id in message_ids:
            email = await asyncio.to_thread(
                fetch_email, context.get_gmail_connection(), message_id
            )
            if email:
                yield "email", email
                yield "message_id", message_id


class MoveToArchive(BaseNode):
    """
    Moves specified emails to Gmail archive.
    email, gmail, archive
    """

    message_id: str = Field(
        default="",
        description="Message ID to archive",
    )

    async def process(self, context: ProcessingContext) -> bool:
        imap = context.get_gmail_connection()
        imap.select("INBOX")

        # Moving to archive in Gmail is done by removing the INBOX label
        result = imap.store(self.message_id, "-X-GM-LABELS", "\\Inbox")
        return result[0] == "OK"


class AddLabel(BaseNode):
    """
    Adds a label to a Gmail message.
    email, gmail, label
    """

    message_id: str = Field(
        default="",
        description="Message ID to label",
    )

    label: str = Field(
        default="",
        description="Label to add to the message",
    )

    async def process(self, context: ProcessingContext) -> bool:
        if not self.message_id:
            raise ValueError("Message ID is required")

        if not self.label:
            raise ValueError("Label is required")

        imap = context.get_gmail_connection()
        imap.select("INBOX")

        result = imap.store(self.message_id, "+X-GM-LABELS", self.label)
        return result[0] == "OK"


class SendEmail(BaseNode):
    """Send a plain text email via SMTP.
    email, smtp, send

    Use cases:
    - Send simple notification messages
    - Automate email reports
    """

    smtp_server: str = Field(
        default="smtp.gmail.com",
        description="SMTP server hostname",
    )
    smtp_port: int = Field(
        default=587,
        description="SMTP server port",
    )
    username: str = Field(
        default="",
        description="SMTP username",
    )
    password: str = Field(
        default="",
        description="SMTP password",
    )
    from_address: str = Field(
        default="",
        description="Sender email address",
    )
    to_address: str = Field(
        default="",
        description="Recipient email address",
    )
    subject: str = Field(
        default="",
        description="Email subject",
    )
    body: str = Field(
        default="",
        description="Email body",
    )

    async def process(self, context: ProcessingContext) -> bool:
        import smtplib
        from email.message import EmailMessage

        if not self.to_address:
            raise ValueError("Recipient email address is required")

        sender = self.from_address or self.username

        msg = EmailMessage()
        msg["Subject"] = self.subject
        msg["From"] = sender
        msg["To"] = self.to_address
        msg.set_content(self.body)

        def _send():
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as smtp:
                smtp.starttls()
                if self.username:
                    smtp.login(self.username, self.password)
                smtp.send_message(msg)

        await asyncio.to_thread(_send)
        return True


def get_date_condition(date_filter: GmailSearch.DateFilter) -> DateSearchCondition:
    """
    Creates a DateSearchCondition based on the specified DateFilter.

    Args:
        date_filter: The DateFilter enum value to convert

    Returns:
        DateSearchCondition configured for the specified filter
    """
    date_deltas = {
        GmailSearch.DateFilter.SINCE_ONE_HOUR: timedelta(hours=1),
        GmailSearch.DateFilter.SINCE_ONE_DAY: timedelta(days=1),
        GmailSearch.DateFilter.SINCE_ONE_WEEK: timedelta(weeks=1),
        GmailSearch.DateFilter.SINCE_ONE_MONTH: timedelta(days=30),
        GmailSearch.DateFilter.SINCE_ONE_YEAR: timedelta(days=365),
    }

    delta = date_deltas[date_filter]
    return DateSearchCondition(
        criteria=DateCriteria.SINCE,
        date=Datetime.from_datetime(datetime.now() - delta),
    )
