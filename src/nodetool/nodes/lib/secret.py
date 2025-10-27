from nodetool.config.environment import Environment
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field


class GetSecret(BaseNode):
    """
    Get a secret value from configuration.
    secrets, credentials, configuration
    """

    name: str = Field(default="", description="Secret key name")
    default: str | None = Field(
        default=None,
        description="Default value if not found",
    )

    async def process(self, context: ProcessingContext) -> str | None:
        return await context.get_secret(self.name) or self.default

