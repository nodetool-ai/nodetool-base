from pathlib import Path
import importlib.util
import pytest
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.common.environment import Environment

SMS_PATH = Path(__file__).resolve().parents[2] / "src" / "nodetool" / "nodes" / "twilio" / "sms.py"
spec = importlib.util.spec_from_file_location("nodetool.nodes.twilio.sms", SMS_PATH)
sms_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sms_module)
SendSMS = sms_module.SendSMS


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_send_sms(context, monkeypatch):
    node = SendSMS(to_number="+1234567890", from_number="+1987654321", body="Hello")

    class FakeResponse:
        def json(self):
            return {"sid": "SM123"}

    async def fake_http_post(url, data=None, headers=None):
        assert data["To"] == "+1234567890"
        assert data["From"] == "+1987654321"
        assert data["Body"] == "Hello"
        return FakeResponse()

    fake_env = {"TWILIO_ACCOUNT_SID": "ACID", "TWILIO_AUTH_TOKEN": "TOKEN"}
    monkeypatch.setattr(Environment, "get_environment", classmethod(lambda cls: fake_env))
    monkeypatch.setattr(context, "http_post", fake_http_post)

    sid = await node.process(context)
    assert sid == "SM123"
