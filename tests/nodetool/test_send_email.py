import pytest
from unittest.mock import patch

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.nodetool.mail import SendEmail


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_send_email(context: ProcessingContext):
    node = SendEmail(
        smtp_server="smtp.example.com",
        smtp_port=587,
        username="user",
        password="pass",
        from_address="from@example.com",
        to_address="to@example.com",
        subject="Hi",
        body="Body",
    )

    with patch("smtplib.SMTP") as mock_smtp:
        instance = mock_smtp.return_value.__enter__.return_value
        await node.process(context)
        instance.starttls.assert_called_once()
        instance.login.assert_called_once_with("user", "pass")
        instance.send_message.assert_called_once()
