import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import AsyncGenerator
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


class GmailSearch(BaseNode):
    """
    Searches Gmail using Gmail-specific search operators.
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

    async def process(self, context: ProcessingContext) -> list[str]:
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

        return search_emails(
            context.get_gmail_connection(), search_criteria, self.max_results
        )


class EmailIterator(BaseNode):
    """
    Iterates over a list of email message IDs.
    email, gmail, iterate
    """

    message_ids: list[str] = Field(
        default=[],
        description="List of message IDs to iterate over",
    )

    @classmethod
    def return_type(cls):
        return {
            "output": Email | None,
        }

    async def gen_process(self, context: ProcessingContext):
        for message_id in self.message_ids:
            yield "output", await asyncio.to_thread(
                fetch_email, context.get_gmail_connection(), message_id
            )


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
        imap = context.get_gmail_connection()
        imap.select("INBOX")

        result = imap.store(self.message_id, "+X-GM-LABELS", self.label)
        return result[0] == "OK"


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
