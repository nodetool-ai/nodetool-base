import pytest
from unittest.mock import MagicMock
from datetime import datetime, timedelta, timezone

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.mail import EmailFields, get_date_condition, GmailSearch, MoveToArchive, AddLabel
from nodetool.metadata.types import Email, Datetime, DateCriteria, DateSearchCondition


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_email_fields_extracts(context: ProcessingContext):
    email = Email(
        id="m-1",
        subject="Subject",
        sender="from@example.com",
        date=Datetime.from_datetime(datetime(2024, 1, 2, 3, 4, 5)),
        body="Hello",
    )
    node = EmailFields(email=email)
    out = await node.process(context)
    assert out["id"] == "m-1"
    assert out["subject"] == "Subject"
    assert out["sender"] == "from@example.com"
    assert out["date"].to_datetime().year == 2024
    assert out["body"] == "Hello"


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
    context.get_gmail_connection = MagicMock(return_value=imap)

    # Archive
    arch = MoveToArchive(message_id="123")
    assert await arch.process(context) is True

    # AddLabel validation
    with pytest.raises(ValueError):
        await AddLabel(message_id="", label="L").process(context)
    with pytest.raises(ValueError):
        await AddLabel(message_id="id", label="").process(context)

    # AddLabel success
    add = AddLabel(message_id="id", label="Label")
    assert await add.process(context) is True

