import asyncio
import random
import imaplib
import socket
import ssl
from typing import ClassVar
from typing import (
    Callable,
    Awaitable,
    Optional,
    TypeVar,
    Tuple,
    Type,
    AsyncGenerator,
    TypedDict,
)
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
    Email,
    EmailSearchCriteria,
)


KEYWORD_SEPARATOR_REGEX = r"\s+|,|;"


T = TypeVar("T")


async def run_with_retries(
    fn: Callable[[], T] | Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 5.0,
    factor: float = 2.0,
    jitter: float = 0.1,
    run_in_thread: bool = True,
    retry_exceptions: Optional[Tuple[Type[BaseException], ...]] = None,
    on_retry: Optional[Callable[[int, Exception], Awaitable[None] | None]] = None,
) -> T:
    """Run a callable with exponential backoff retries.

    Args:
        fn: Zero-arg callable to execute.
        attempts: Max attempts before giving up.
        base_delay: Initial delay seconds.
        max_delay: Max backoff delay seconds.
        factor: Exponential factor.
        jitter: Added random jitter [0, jitter] seconds.
        run_in_thread: Execute the function in a worker thread.

    Returns:
        Result of the callable.

    Raises:
        Exception from the final attempt.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            if run_in_thread:
                return await asyncio.to_thread(fn)  # type: ignore[arg-type]
            result = fn()
            if asyncio.iscoroutine(result):  # type: ignore[truthy-function]
                return await result  # type: ignore[misc]
            return result  # type: ignore[return-value]
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if retry_exceptions is not None and not isinstance(exc, retry_exceptions):
                raise
            if attempt >= attempts:
                raise
            if on_retry is not None:
                maybe_coro = on_retry(attempt, exc)
                if asyncio.iscoroutine(maybe_coro):
                    await maybe_coro  # type: ignore[misc]
            delay = min(max_delay, base_delay * (factor ** (attempt - 1)))
            delay += random.uniform(0, max(0.0, jitter))
            await asyncio.sleep(delay)
    # Should not reach here
    assert last_exc is not None
    raise last_exc


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

    class OutputType(TypedDict):
        id: str
        subject: str
        sender: str
        date: Datetime
        body: str

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

    # Retry configuration
    retry_attempts: int = Field(
        default=3,
        description="Maximum retry attempts for Gmail operations",
    )
    retry_base_delay: float = Field(
        default=0.5,
        description="Base delay (seconds) for exponential backoff",
    )
    retry_max_delay: float = Field(
        default=5.0,
        description="Maximum delay (seconds) for exponential backoff",
    )
    retry_factor: float = Field(
        default=2.0,
        description="Exponential growth factor for backoff",
    )
    retry_jitter: float = Field(
        default=0.1,
        description="Random jitter (seconds) added to each backoff",
    )

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["from_address", "subject", "body", "date_filter", "max_results"]

    class OutputType(TypedDict):
        email: Email
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
        gmail_connection = await context.get_gmail_connection()

        async def _on_retry(attempt: int, exc: Exception):
            gmail_connection.shutdown()
            gmail_connection.logout()
            gmail_connection.select("INBOX")

        message_ids = await run_with_retries(
            lambda: search_emails(gmail_connection, search_criteria, self.max_results),
            attempts=self.retry_attempts,
            base_delay=self.retry_base_delay,
            max_delay=self.retry_max_delay,
            factor=self.retry_factor,
            jitter=self.retry_jitter,
            run_in_thread=True,
            retry_exceptions=(
                imaplib.IMAP4.abort,
                socket.timeout,
                ssl.SSLError,
                OSError,
            ),
            on_retry=_on_retry,
        )

        for message_id in message_ids:
            email = await run_with_retries(
                lambda: fetch_email(gmail_connection, message_id),
                attempts=self.retry_attempts,
                base_delay=self.retry_base_delay,
                max_delay=self.retry_max_delay,
                factor=self.retry_factor,
                jitter=self.retry_jitter,
                run_in_thread=True,
                retry_exceptions=(
                    imaplib.IMAP4.abort,
                    socket.timeout,
                    ssl.SSLError,
                    OSError,
                ),
                on_retry=_on_retry,
            )
            if email:
                yield {"email": email, "message_id": message_id}


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

    async def process(self, context: ProcessingContext) -> bool:
        attempts = 3
        base_delay = 0.5
        max_delay = 5.0
        factor = 2.0
        jitter = 0.1
        gmail_connection = await context.get_gmail_connection()

        async def _on_retry(attempt: int, exc: Exception):
            gmail_connection.shutdown()
            gmail_connection.logout()
            gmail_connection.select("INBOX")

        def op() -> bool:
            gmail_connection.select("INBOX")
            result = gmail_connection.store(self.message_id, "-X-GM-LABELS", "\\Inbox")
            if result[0] != "OK":
                raise OSError(f"IMAP STORE archive failed: {result}")
            return True

        return await run_with_retries(
            op,
            attempts=attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            factor=factor,
            jitter=jitter,
            run_in_thread=True,
            retry_exceptions=(
                imaplib.IMAP4.abort,
                socket.timeout,
                ssl.SSLError,
                OSError,
            ),
            on_retry=_on_retry,
        )


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

    async def process(self, context: ProcessingContext) -> bool:
        if not self.message_id:
            raise ValueError("Message ID is required")

        if not self.label:
            raise ValueError("Label is required")

        attempts = 3
        base_delay = 0.5
        max_delay = 5.0
        factor = 2.0
        jitter = 0.1

        gmail_connection = await context.get_gmail_connection()

        async def _on_retry(attempt: int, exc: Exception):
            gmail_connection.shutdown()
            gmail_connection.logout()
            gmail_connection.select("INBOX")

        def op() -> bool:
            gmail_connection.select("INBOX")
            result = gmail_connection.store(self.message_id, "-X-GM-LABELS", "\\Inbox")
            if result[0] != "OK":
                raise OSError(f"IMAP STORE archive failed: {result}")
            return True

        return await run_with_retries(
            op,
            attempts=attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            factor=factor,
            jitter=jitter,
            run_in_thread=True,
            retry_exceptions=(
                imaplib.IMAP4.abort,
                socket.timeout,
                ssl.SSLError,
                OSError,
            ),
            on_retry=_on_retry,
        )


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

    # Retry configuration
    retry_attempts: int = Field(
        default=3,
        description="Maximum retry attempts for SMTP send",
    )
    retry_base_delay: float = Field(
        default=0.5,
        description="Base delay (seconds) for exponential backoff",
    )
    retry_max_delay: float = Field(
        default=5.0,
        description="Maximum delay (seconds) for exponential backoff",
    )
    retry_factor: float = Field(
        default=2.0,
        description="Exponential growth factor for backoff",
    )
    retry_jitter: float = Field(
        default=0.1,
        description="Random jitter (seconds) added to each backoff",
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

        await run_with_retries(
            _send,
            attempts=self.retry_attempts,
            base_delay=self.retry_base_delay,
            max_delay=self.retry_max_delay,
            factor=self.retry_factor,
            jitter=self.retry_jitter,
            run_in_thread=True,
        )
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
