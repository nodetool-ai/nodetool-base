import fnmatch
from typing import Any, AsyncGenerator, TypedDict, ClassVar
from nodetool.config.environment import Environment
from nodetool.metadata.types import FolderRef, ImageModel, Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import ImageRef
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import create_file_uri
from nodetool.providers.types import TextToImageParams, ImageToImageParams
from nodetool.workflows.types import SaveUpdate
import os
import datetime
from pydantic import Field
import PIL
import PIL.Image


class LoadImageFile(BaseNode):
    """
    Read an image file from disk.
    image, input, load, file
    """

    path: str = Field(default="", description="Path to the image file to read")

    async def process(self, context: ProcessingContext) -> ImageRef:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.path:
            raise ValueError("path cannot be empty")

        expanded_path = os.path.expanduser(self.path)
        if not os.path.exists(expanded_path):
            raise ValueError(f"Image file not found: {expanded_path}")

        with open(expanded_path, "rb") as f:
            image_data = f.read()

        image = await context.image_from_bytes(image_data)
        image.uri = create_file_uri(expanded_path)
        return image


class LoadImageFolder(BaseNode):
    """
    Load all images from a folder, optionally including subfolders.
    image, load, folder, files
    """

    folder: str = Field(default="", description="Folder to scan for images")
    include_subdirectories: bool = Field(
        default=False, description="Include images in subfolders"
    )
    extensions: list[str] = Field(
        default=[".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"],
        description="Image file extensions to include",
    )
    pattern: str = Field(default="", description="Pattern to match image files")

    @classmethod
    def get_title(cls):
        return "Load Image Folder"

    class OutputType(TypedDict):
        image: ImageRef
        path: str

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.folder:
            raise ValueError("folder cannot be empty")

        expanded_folder = os.path.expanduser(self.folder)
        if not os.path.isdir(expanded_folder):
            raise ValueError(f"Folder does not exist: {expanded_folder}")

        allowed_exts = {ext.lower() for ext in self.extensions}

        def iter_files(base_folder: str):
            if self.include_subdirectories:
                for root, _, files in os.walk(base_folder):
                    for f in files:
                        yield os.path.join(root, f)
            else:
                for f in os.listdir(base_folder):
                    yield os.path.join(base_folder, f)

        for path in iter_files(expanded_folder):
            if not os.path.isfile(path):
                continue
            _, ext = os.path.splitext(path)
            if ext.lower() not in allowed_exts:
                continue

            if self.pattern and not fnmatch.fnmatch(path, self.pattern):
                continue

            with open(path, "rb") as f:
                image_data = f.read()

            image = await context.image_from_bytes(image_data)
            image.uri = create_file_uri(path)
            yield {"path": path, "image": image}


class SaveImageFile(BaseNode):
    """
    Write an image to disk.
    image, output, save, file
    """

    image: ImageRef = Field(default=ImageRef(), description="The image to save")
    folder: str = Field(default="", description="Folder where the file will be saved")
    filename: str = Field(
        default="",
        description="""
        The name of the image file.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
        """,
    )
    overwrite: bool = Field(
        default=False,
        description="Overwrite the file if it already exists, otherwise file will be renamed",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.folder:
            raise ValueError("folder cannot be empty")
        if not self.filename:
            raise ValueError("filename cannot be empty")

        expanded_folder = os.path.expanduser(self.folder)
        if not os.path.exists(expanded_folder):
            raise ValueError(f"Folder does not exist: {expanded_folder}")

        filename = datetime.datetime.now().strftime(self.filename)

        expanded_path = os.path.join(expanded_folder, filename)
        os.makedirs(os.path.dirname(expanded_path), exist_ok=True)

        image = await context.image_to_pil(self.image)
        if not self.overwrite:
            count = 1
            while os.path.exists(expanded_path):
                fname, ext = os.path.splitext(filename)
                filename = f"{fname}_{count}{ext}"
                expanded_path = os.path.join(expanded_folder, filename)
                count += 1
        image.save(expanded_path)
        result = ImageRef(uri=create_file_uri(expanded_path), data=image.tobytes())

        # Emit SaveUpdate event
        context.post_message(
            SaveUpdate(
                node_id=self.id, name=filename, value=result, output_type="image"
            )
        )

        return result


class LoadImageAssets(BaseNode):
    """
    Load images from an asset folder.
    load, image, file, import
    """

    folder: FolderRef = Field(
        default=FolderRef(), description="The asset folder to load the images from."
    )

    @classmethod
    def get_title(cls):
        return "Load Image Assets"

    class OutputType(TypedDict):
        image: ImageRef
        name: str

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        if self.folder.is_empty():
            raise ValueError("Please select an asset folder.")

        parent_id = self.folder.asset_id
        list_assets, _ = await context.list_assets(
            parent_id=parent_id, content_type="image"
        )

        for asset in list_assets:
            yield {
                "name": asset.name,
                "image": ImageRef(
                    type="image",
                    uri=await context.get_asset_url(asset.id),
                    asset_id=asset.id,
                ),
            }


class SaveImage(BaseNode):
    """
    Save an image to specified asset folder with customizable name format.
    save, image, folder, naming
    """

    image: ImageRef = Field(default=ImageRef(), description="The image to save.")
    folder: FolderRef = Field(
        default=FolderRef(), description="The asset folder to save the image in."
    )
    name: str = Field(
        default="%Y-%m-%d_%H-%M-%S.png",
        description="""
        Name of the output file.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
        """,
    )

    @classmethod
    def get_title(cls):
        return "Save Image Asset"

    def required_inputs(self):
        return ["image"]

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self.image.is_empty():
            raise ValueError("The input image is not connected.")

        image = await context.image_to_pil(self.image)
        filename = datetime.datetime.now().strftime(self.name)
        parent_id = self.folder.asset_id if self.folder.is_set() else None

        result = await context.image_from_pil(
            image=image, name=filename, parent_id=parent_id
        )

        # Emit SaveUpdate event
        context.post_message(
            SaveUpdate(
                node_id=self.id, name=filename, value=result, output_type="image"
            )
        )

        return result

    def result_for_client(self, result: dict[str, Any]) -> dict[str, Any]:
        return self.result_for_all_outputs(result)


class GetMetadata(BaseNode):
    """
    Get metadata about the input image.
    metadata, properties, analysis, information
    """

    image: ImageRef = Field(default=ImageRef(), description="The input image.")

    class OutputType(TypedDict):
        format: str
        mode: str
        width: int
        height: int
        channels: int

    async def process(self, context: ProcessingContext) -> OutputType:
        if self.image.is_empty():
            raise ValueError("The input image is not connected.")

        image = await context.image_to_pil(self.image)

        # Get basic image information
        format = image.format if image.format else "Unknown"
        mode = image.mode
        width, height = image.size
        channels = len(image.getbands())

        return {
            "format": format,
            "mode": mode,
            "width": width,
            "height": height,
            "channels": channels,
        }


class BatchToList(BaseNode):
    """
    Convert an image batch to a list of image references.
    batch, list, images, processing
    """

    batch: ImageRef = Field(
        default=ImageRef(), description="The batch of images to convert."
    )

    async def process(self, context: ProcessingContext) -> list[ImageRef]:
        if self.batch.is_empty():
            raise ValueError("The input batch is not connected.")
        if self.batch.data is None:
            raise ValueError("The input batch is empty.")
        if not isinstance(self.batch.data, list):
            raise ValueError("The input batch is not a list.")

        return [ImageRef(data=data) for data in self.batch.data]


class ImagesToList(BaseNode):
    """
    Convert all dynamic properties to a list of image references.
    list, images, processing
    """

    _is_dynamic = True

    async def process(self, context: ProcessingContext) -> list[ImageRef]:
        if not self.dynamic_properties:
            return []

        results = []
        for data in self.dynamic_properties.values():
            if isinstance(data, ImageRef):
                results.append(data)
            else:
                results.append(ImageRef(data=data))
        return results


class Paste(BaseNode):
    """
    Paste one image onto another at specified coordinates.
    paste, composite, positioning, overlay
    """

    image: ImageRef = Field(default=ImageRef(), description="The image to paste into.")
    paste: ImageRef = Field(default=ImageRef(), description="The image to paste.")
    left: int = Field(default=0, ge=0, le=4096, description="The left coordinate.")
    top: int = Field(default=0, ge=0, le=4096, description="The top coordinate.")

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self.image.is_empty():
            raise ValueError("The input image is not connected.")
        if self.paste.is_empty():
            raise ValueError("The paste image is not connected.")

        image = await context.image_to_pil(self.image)
        paste = await context.image_to_pil(self.paste)
        image.paste(paste, (self.left, self.top))
        return await context.image_from_pil(image)


class Scale(BaseNode):
    """
    Enlarge or shrink an image by a scale factor.
    image, resize, scale
    """

    image: ImageRef = Field(default=ImageRef(), description="The image to scale.")
    scale: float = Field(default=1.0, ge=0.0, le=10.0, description="The scale factor.")

    async def process(self, context: ProcessingContext) -> ImageRef:
        image = await context.image_to_pil(self.image)
        width = int((image.width * self.scale))
        height = int((image.height * self.scale))
        image = image.resize((width, height), PIL.Image.Resampling.LANCZOS)
        return await context.image_from_pil(image)


class Resize(BaseNode):
    """
    Change image dimensions to specified width and height.
    image, resize
    """

    image: ImageRef = Field(default=ImageRef(), description="The image to resize.")
    width: int = Field(default=512, ge=0, le=4096, description="The target width.")
    height: int = Field(default=512, ge=0, le=4096, description="The target height.")

    async def process(self, context: ProcessingContext) -> ImageRef:
        image = await context.image_to_pil(self.image)
        res = image.resize((self.width, self.height), PIL.Image.LANCZOS)  # type: ignore
        return await context.image_from_pil(res)


class Crop(BaseNode):
    """
    Crop an image to specified coordinates.
    image, crop
    """

    image: ImageRef = Field(default=ImageRef(), description="The image to crop.")
    left: int = Field(default=0, ge=0, le=4096, description="The left coordinate.")
    top: int = Field(default=0, ge=0, le=4096, description="The top coordinate.")
    right: int = Field(default=512, ge=0, le=4096, description="The right coordinate.")
    bottom: int = Field(
        default=512, ge=0, le=4096, description="The bottom coordinate."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image = await context.image_to_pil(self.image)
        res = image.crop((self.left, self.top, self.right, self.bottom))
        return await context.image_from_pil(res)


class Fit(BaseNode):
    """
    Resize an image to fit within specified dimensions while preserving aspect ratio.
    image, resize, fit
    """

    image: ImageRef = Field(default=ImageRef(), description="The image to fit.")
    width: int = Field(default=512, ge=1, le=4096, description="Width to fit to.")
    height: int = Field(default=512, ge=1, le=4096, description="Height to fit to.")

    async def process(self, context: ProcessingContext) -> ImageRef:
        image = await context.image_to_pil(self.image)
        res = PIL.ImageOps.fit(image, (self.width, self.height), PIL.Image.LANCZOS)  # type: ignore
        return await context.image_from_pil(res)


class TextToImage(BaseNode):
    """
    Generate images from text prompts using any supported image provider. Automatically routes to the appropriate backend (HuggingFace, FAL, MLX).
    image, generation, AI, text-to-image, t2i
    """

    _expose_as_tool: ClassVar[bool] = True

    model: ImageModel = Field(
        default=ImageModel(
            provider=Provider.HuggingFaceFalAI,
            id="fal-ai/flux/schnell",
            name="FLUX.1 Schnell",
        ),
        description="The image generation model to use",
    )
    prompt: str = Field(
        default="A cat holding a sign that says hello world",
        description="Text prompt describing the desired image",
    )
    negative_prompt: str = Field(
        default="",
        description="Text prompt describing what to avoid in the image",
    )
    width: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Width of the generated image",
    )
    height: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Height of the generated image",
    )
    guidance_scale: float = Field(
        default=7.5,
        ge=0.0,
        le=30.0,
        description="Classifier-free guidance scale (higher = closer to prompt)",
    )
    num_inference_steps: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Number of denoising steps",
    )
    seed: int = Field(
        default=-1,
        ge=-1,
        description="Random seed for reproducibility (-1 for random)",
    )
    safety_check: bool = Field(
        default=True,
        description="Enable safety checker to filter inappropriate content",
    )
    timeout_seconds: int = Field(
        default=0,
        ge=0,
        le=3600,
        description="Timeout in seconds for API calls (0 = use provider default)",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        # Get the image provider for this model
        provider_instance = await context.get_provider(self.model.provider)

        params = TextToImageParams(
            model=self.model,
            prompt=self.prompt,
            negative_prompt=self.negative_prompt if self.negative_prompt else None,
            width=self.width,
            height=self.height,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            seed=self.seed if self.seed != -1 else None,
            safety_check=self.safety_check,
        )

        # Generate image
        image_bytes = await provider_instance.text_to_image(
            params,
            timeout_s=self.timeout_seconds if self.timeout_seconds > 0 else None,
            context=context,
            node_id=self.id,
        )

        # Convert to ImageRef
        return await context.image_from_bytes(image_bytes)

    @classmethod
    def get_basic_fields(cls):
        return ["model", "prompt", "width", "height", "seed"]


class ImageToImage(BaseNode):
    """
    Transform images using text prompts with any supported image provider. Automatically routes to the appropriate backend (HuggingFace, FAL, MLX).
    image, transformation, AI, image-to-image, i2i
    """

    _expose_as_tool: ClassVar[bool] = True

    model: ImageModel = Field(
        default=ImageModel(
            provider=Provider.HuggingFaceFalAI,
            id="fal-ai/flux/dev",
            name="FLUX.1 Dev",
        ),
        description="The image generation model to use",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        description="Input image to transform",
    )
    prompt: str = Field(
        default="A photorealistic version of the input image",
        description="Text prompt describing the desired transformation",
    )
    negative_prompt: str = Field(
        default="",
        description="Text prompt describing what to avoid",
    )
    strength: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="How much to transform the input image (0.0 = no change, 1.0 = maximum change)",
    )
    guidance_scale: float = Field(
        default=7.5,
        ge=0.0,
        le=30.0,
        description="Classifier-free guidance scale",
    )
    num_inference_steps: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Number of denoising steps",
    )
    target_width: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Target width of the output image",
    )
    target_height: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Target height of the output image",
    )
    seed: int = Field(
        default=-1,
        ge=-1,
        description="Random seed for reproducibility (-1 for random)",
    )
    scheduler: str = Field(
        default="",
        description="Scheduler to use (provider-specific)",
    )
    safety_check: bool = Field(
        default=True,
        description="Enable safety checker",
    )
    timeout_seconds: int = Field(
        default=0,
        ge=0,
        le=3600,
        description="Timeout in seconds for API calls (0 = use provider default)",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self.image.is_empty():
            raise ValueError("Input image is required")

        # Get the image provider for this model
        provider_instance = await context.get_provider(self.model.provider)

        # Get input image bytes
        input_image_bytes = await context.asset_to_bytes(self.image)
        params = ImageToImageParams(
            model=self.model,
            prompt=self.prompt,
            negative_prompt=self.negative_prompt if self.negative_prompt else None,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            strength=self.strength,
            target_width=self.target_width,
            target_height=self.target_height,
            seed=self.seed if self.seed != -1 else None,
            scheduler=self.scheduler if self.scheduler else None,
        )

        # Transform image
        output_image_bytes = await provider_instance.image_to_image(
            image=input_image_bytes,
            params=params,
            timeout_s=self.timeout_seconds if self.timeout_seconds > 0 else None,
            context=context,
            node_id=self.id,
        )

        # Convert to ImageRef
        return await context.image_from_bytes(output_image_bytes)

    @classmethod
    def get_basic_fields(cls):
        return ["model", "image", "prompt", "strength", "seed"]
