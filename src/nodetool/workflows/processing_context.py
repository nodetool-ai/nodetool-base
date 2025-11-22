from __future__ import annotations

import os
import tempfile
from io import BytesIO
from typing import Any

from PIL import Image

from nodetool.metadata.types import AudioRef, DocumentRef, ImageRef
from nodetool.io.uri_utils import create_file_uri


class ProcessingContext:
    """A minimal processing context used for tests."""

    def __init__(self, user_id: str | None = None, auth_token: str | None = None):
        self.user_id = user_id
        self.auth_token = auth_token
        self.messages: list[Any] = []

    def post_message(self, message: Any) -> None:
        self.messages.append(message)

    async def download_file(self, uri: str):
        path = uri
        if uri.startswith("file://"):
            path = uri[len("file://") :]
        tmp = tempfile.NamedTemporaryFile(delete=False)
        with open(path, "rb") as src:
            tmp.write(src.read())
        tmp.flush()
        return tmp

    async def asset_to_bytes(self, ref: AudioRef) -> bytes:
        return ref.data or b""

    async def download_uri(self, uri: str):
        return await self.download_file(uri)

    async def image_from_bytes(
        self, data: bytes, name: str | None = None, parent_id: str | None = None
    ) -> ImageRef:
        # Normalize the image by loading and re-saving so downstream size checks work
        image = Image.open(BytesIO(data))
        return await self.image_from_pil(image, name=name, parent_id=parent_id)

    async def image_to_pil(self, image_ref: ImageRef) -> Image.Image:
        if image_ref.data is None:
            raise ValueError("Image data is empty.")
        if isinstance(image_ref.data, list):
            raise ValueError("Image data is a batch; convert to a single image first.")
        return Image.open(BytesIO(image_ref.data))

    async def image_from_pil(
        self, image: Image.Image, name: str | None = None, parent_id: str | None = None
    ) -> ImageRef:
        buf = BytesIO()
        format = image.format or "PNG"
        image.save(buf, format=format)
        return ImageRef(
            type="image",
            data=buf.getvalue(),
            uri=name if name else "",
            asset_id=parent_id,
        )

    async def document_from_bytes(self, data: bytes, name: str | None = None) -> DocumentRef:
        return DocumentRef(uri=name or "", data=data)

    async def get_asset_url(self, asset_id: str) -> str:
        return create_file_uri(asset_id)

    async def get_provider(self, provider: Any) -> Any:
        raise RuntimeError("Provider access is not available in this test context.")

