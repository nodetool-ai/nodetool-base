from __future__ import annotations

from base64 import b64encode

from pydantic import Field

from nodetool.common.environment import Environment
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class SendSMS(BaseNode):
    """Send an SMS message using the Twilio API.

    twilio, sms, message, send

    Use cases:
    - Notify via text messages
    - Two-factor authentication
    - Alerts and reminders
    """

    to_number: str = Field(default="", description="Destination phone number in E.164 format")
    from_number: str = Field(default="", description="Twilio phone number to send from")
    body: str = Field(default="", description="Message body to send")

    async def process(self, context: ProcessingContext) -> str:
        if not self.to_number:
            raise ValueError("Destination phone number is required")
        if not self.body:
            raise ValueError("Message body is required")

        env = Environment.get_environment()
        account_sid = env.get("TWILIO_ACCOUNT_SID")
        auth_token = env.get("TWILIO_AUTH_TOKEN")
        from_number = self.from_number or env.get("TWILIO_PHONE_NUMBER", "")

        if not account_sid or not auth_token:
            raise ValueError("Twilio credentials not configured")
        if not from_number:
            raise ValueError("Twilio sender phone number not configured")

        url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
        auth_header = b64encode(f"{account_sid}:{auth_token}".encode()).decode()
        headers = {"Authorization": f"Basic {auth_header}"}
        data = {
            "To": self.to_number,
            "From": from_number,
            "Body": self.body,
        }

        res = await context.http_post(url, data=data, headers=headers)
        json_res = res.json()
        return json_res.get("sid", "")
