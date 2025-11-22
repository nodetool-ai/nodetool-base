import os


def create_file_uri(path: str) -> str:
    """Create a file:// URI for a local path."""
    return f"file://{os.path.abspath(path)}"

