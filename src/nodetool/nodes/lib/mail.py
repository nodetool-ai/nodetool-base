import imaplib
import asyncio
import random
from typing import ClassVar, TypeVar, AsyncGenerator, TypedDict, Any
from functools import partial
from datetime import datetime, timedelta
from enum import Enum
from pydantic import Field
from nodetool.system.imap import search_emails, fetch_email
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import (
    DateCriteria,
    DateSearchCondition,
    Datetime,
    EmailSearchCriteria,
)
import logging

logger = logging.getLogger(__name__)


KEYWORD_SEPARATOR_REGEX = r"\s+|,|;"


T = TypeVar("T")


async def get_gmail_connection(user_id: str) -> imaplib.IMAP4_SSL:
    """
    Creates a Gmail connection configuration.

    Args:
        user_id: User ID to connect to

    Returns:
        IMAPConnection configured for Gmail

    Raises:
        ValueError: If email_address or app_password is empty
    """
    from nodetool.security.secret_helper import get_secret_required

    email_address = await get_secret_required("GOOGLE_MAIL_USER", user_id)
    app_password = await get_secret_required("GOOGLE_APP_PASSWORD", user_id)

    imap = imaplib.IMAP4_SSL("imap.gmail.com", 993)
    imap.login(email_address, app_password)
    return imap

async def run_with_retries(
    fn: Any,
    *args: Any,
    attempts: int = 3,
    delay: float = 1.0,
    **kwargs: Any,
) -> Any:
    """Run a sync function in a thread with exponential backoff retries.

    Args:
        fn: Sync function to execute in a thread.
        *args: Positional arguments to pass to fn.
        attempts: Max attempts before giving up.
        delay: Base delay in seconds (doubles each retry, with 10% jitter).
        **kwargs: Keyword arguments to pass to fn.

    Returns:
        Result of the function.

    Raises:
        The exception from the final attempt.
    """
    for attempt in range(attempts):
        try:
            return await asyncio.to_thread(partial(fn, *args, **kwargs))
        except Exception:
            logger.exception("Failed to run function with retries")
            if attempt >= attempts - 1:
                raise
            await asyncio.sleep(delay * (2**attempt) * (1 + random.uniform(0, 0.1)))
    raise RuntimeError("Unreachable")


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

    _expose_as_tool: ClassVar[bool] = True
    _required_settings: ClassVar[list[str]] = ["GOOGLE_MAIL_USER", "GOOGLE_APP_PASSWORD"]

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["from_address", "subject", "body", "date_filter", "max_results"]

    class OutputType(TypedDict):
        email: dict
        message_id: str

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
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
        gmail_connection = await get_gmail_connection(context.user_id)

        message_ids = await run_with_retries(
            search_emails, gmail_connection, search_criteria, self.max_results
        )

        for message_id in message_ids:
            email = await run_with_retries(fetch_email, gmail_connection, message_id)
            if email:
                yield {
                    "email": email,
                    "message_id": message_id,
                }


class MoveToArchive(BaseNode):
    """
    Moves specified emails to Gmail archive.
    email, gmail, archive
    """

    message_id: str = Field(
        default="",
        description="Message ID to archive",
    )

    _expose_as_tool: ClassVar[bool] = True
    _required_settings: ClassVar[list[str]] = ["GOOGLE_MAIL_USER", "GOOGLE_APP_PASSWORD"]

    async def process(self, context: ProcessingContext) -> bool:
        gmail_connection = await get_gmail_connection(context.user_id)

        def archive_email() -> bool:
            gmail_connection.select("INBOX")
            result = gmail_connection.store(self.message_id, "-X-GM-LABELS", "\\Inbox")
            if result[0] != "OK":
                raise OSError(f"IMAP STORE archive failed: {result}")
            return True

        return await run_with_retries(archive_email)


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

    _expose_as_tool: ClassVar[bool] = True
    _required_settings: ClassVar[list[str]] = ["GOOGLE_MAIL_USER", "GOOGLE_APP_PASSWORD"]

    async def process(self, context: ProcessingContext) -> bool:
        if not self.message_id:
            raise ValueError("Message ID is required")
        if not self.label:
            raise ValueError("Label is required")

        gmail_connection = await get_gmail_connection(context.user_id)

        def add_label() -> bool:
            gmail_connection.select("INBOX")
            result = gmail_connection.store(self.message_id, "+X-GM-LABELS", self.label)
            if result[0] != "OK":
                raise OSError(f"IMAP STORE label failed: {result}")
            return True

        return await run_with_retries(add_label)


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

    _expose_as_tool: ClassVar[bool] = True

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

        def send():
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as smtp:
                smtp.starttls()
                if self.username:
                    smtp.login(self.username, self.password)
                smtp.send_message(msg)

        await run_with_retries(send)
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
