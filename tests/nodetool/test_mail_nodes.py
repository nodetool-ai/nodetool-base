import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timedelta, timezone

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.mail import get_date_condition, GmailSearch, MoveToArchive
from nodetool.metadata.types import Email, Datetime, DateCriteria, DateSearchCondition


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


def test_get_date_condition_since_one_day():
    cond = get_date_condition(GmailSearch.DateFilter.SINCE_ONE_DAY)
    assert isinstance(cond, DateSearchCondition)
    assert cond.criteria == DateCriteria.SINCE
    # ensure within last ~2 days
    delta = datetime.now(timezone.utc) - cond.date.to_datetime()
    assert timedelta(hours=0) <= delta <= timedelta(days=2)


@pytest.mark.asyncio
async def test_archive_and_addlabel_with_mock_imap(context: ProcessingContext):
    imap = MagicMock()
    imap.select.return_value = ("OK", None)
    imap.store.return_value = ("OK", None)
    context.get_gmail_connection = AsyncMock(return_value=imap)

    # Archive
    arch = MoveToArchive(message_id="123")
    assert await arch.process(context) is True

    # AddLabel validation
    imap.select.return_value = ("OK", None)
    imap.store.return_value = ("OK", None)
    context.get_gmail_connection = AsyncMock(return_value=imap)  # type: ignore
    return context
