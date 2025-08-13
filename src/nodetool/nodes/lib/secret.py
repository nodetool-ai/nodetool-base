from nodetool.common.environment import Environment
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
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        return Environment.get(self.name, self.default)


class SetSecret(BaseNode):
    """
    Set a secret value and persist it.
    secrets, credentials, configuration
    """

    name: str = Field(default="", description="Secret key name")
    value: str = Field(default="", description="Secret value")

    async def process(self, context: ProcessingContext) -> None:
        from nodetool.common.settings import save_settings

        if Environment.is_production():
            raise ValueError("This node is not available in production")

        settings = Environment.get_settings()
        secrets = Environment.get_secrets()
        secrets[self.name] = self.value
        save_settings(settings, secrets)
        return None
