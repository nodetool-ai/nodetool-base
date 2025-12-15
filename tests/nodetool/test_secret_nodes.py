import pytest
import pytest_asyncio
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.secret import GetSecret
from nodetool.models.secret import Secret


@pytest.fixture
def mock_keyring(monkeypatch):
    """Mock the keyring to avoid accessing the system keychain."""
    # Create a simple in-memory store for the master key
    keyring_store = {}

    def mock_get_password(service, username):
        return keyring_store.get((service, username))

    def mock_set_password(service, username, password):
        keyring_store[(service, username)] = password

    def mock_delete_password(service, username):
        keyring_store.pop((service, username), None)

    monkeypatch.setattr("keyring.get_password", mock_get_password)
    monkeypatch.setattr("keyring.set_password", mock_set_password)
    monkeypatch.setattr("keyring.delete_password", mock_delete_password)

    # Clear the cached master key before test
    from nodetool.security.master_key import MasterKeyManager
    MasterKeyManager.clear_cache()

    yield keyring_store

    # Clean up after test
    MasterKeyManager.clear_cache()


@pytest_asyncio.fixture
async def context(tmp_path, monkeypatch, mock_keyring):
    """Set up test context with mocked keyring and database."""
    monkeypatch.setenv("HOME", str(tmp_path))

    # Initialize the database table for secrets
    await Secret.create_table()

    return ProcessingContext(user_id="test", auth_token="test")


@pytest.mark.asyncio
async def test_secret_nodes(context: ProcessingContext):
    # Create secret for the test user
    await Secret.create(user_id=context.user_id, key="TEST_SECRET", value="value")

    get_node = GetSecret(name="TEST_SECRET")
    result = await get_node.process(context)
    assert result == "value"
