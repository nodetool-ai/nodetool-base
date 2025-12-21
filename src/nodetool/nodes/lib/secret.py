from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field


class GetSecret(BaseNode):
    """
    Retrieve secret value from secure configuration storage.

    Fetches a secret by name from the configured secrets backend. Returns the
    secret value or a default if not found. Secrets are not exposed in logs.

    Parameters:
    - name (required): Secret key identifier
    - default (optional, default=""): Value returned if secret not found

    Returns: Secret value as string, or default

    Side effects: Accesses secrets backend

    Typical usage: Retrieve API keys, tokens, passwords, or credentials for use in
    HTTP requests, database connections, or API nodes. Never hardcode secrets; always
    use this node. Follow with nodes that require authentication.

    secrets, credentials, configuration
    """

    name: str = Field(default="", description="Secret key name")
    default: str = Field(
        default="",
        description="Default value if not found",
    )

    async def process(self, context: ProcessingContext) -> str | None:
        return await context.get_secret(self.name) or self.default

