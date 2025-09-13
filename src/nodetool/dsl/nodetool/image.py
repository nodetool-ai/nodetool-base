from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class BatchToList(GraphNode):
    """
    Convert an image batch to a list of image references.
    batch, list, images, processing

    Use cases:
    - Convert comfy batch outputs to list format
    """

    batch: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The batch of images to convert.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.image.BatchToList"


class Crop(GraphNode):
    """
    Crop an image to specified coordinates.
    image, crop

    - Remove unwanted borders from images
    - Focus on particular subjects within an image
    - Simplify images by removing distractions
    """

    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The image to crop.",
    )
    left: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="The left coordinate."
    )
    top: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="The top coordinate."
    )
    right: int | GraphNode | tuple[GraphNode, str] = Field(
        default=512, description="The right coordinate."
    )
    bottom: int | GraphNode | tuple[GraphNode, str] = Field(
        default=512, description="The bottom coordinate."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.image.Crop"


class Fit(GraphNode):
    """
    Resize an image to fit within specified dimensions while preserving aspect ratio.
    image, resize, fit

    - Resize images for online publishing requirements
    - Preprocess images to uniform sizes for machine learning
    - Control image display sizes for web development
    """

    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The image to fit.",
    )
    width: int | GraphNode | tuple[GraphNode, str] = Field(
        default=512, description="Width to fit to."
    )
    height: int | GraphNode | tuple[GraphNode, str] = Field(
        default=512, description="Height to fit to."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.image.Fit"


class GetMetadata(GraphNode):
    """
    Get metadata about the input image.
    metadata, properties, analysis, information

    Use cases:
    - Use width and height for layout calculations
    - Analyze image properties for processing decisions
    - Gather information for image cataloging or organization
    """

    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The input image.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.image.GetMetadata"


class LoadImageAssets(GraphNode):
    """
    Load images from an asset folder.
    load, image, file, import
    """

    folder: types.FolderRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FolderRef(type="folder", uri="", asset_id=None, data=None),
        description="The asset folder to load the images from.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.image.LoadImageAssets"


class LoadImageFile(GraphNode):
    """
    Read an image file from disk.
    image, input, load, file

    Use cases:
    - Load images for processing
    - Import photos for editing
    - Read image assets for a workflow
    """

    path: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path=""),
        description="Path to the image file to read",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.image.LoadImageFile"


class LoadImageFolder(GraphNode):
    """
    Load all images from a folder, optionally including subfolders.
    image, load, folder, files

    Use cases:
    - Batch import images for processing
    - Build datasets from a directory tree
    - Iterate over photo collections
    """

    folder: types.FolderPath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FolderPath(type="folder_path", path=""),
        description="Folder to scan for images",
    )
    include_subdirectories: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False, description="Include images in subfolders"
    )
    extensions: list[str] | GraphNode | tuple[GraphNode, str] = Field(
        default=[".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"],
        description="Image file extensions to include",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.image.LoadImageFolder"


class Paste(GraphNode):
    """
    Paste one image onto another at specified coordinates.
    paste, composite, positioning, overlay

    Use cases:
    - Add watermarks or logos to images
    - Combine multiple image elements
    - Create collages or montages
    """

    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The image to paste into.",
    )
    paste: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The image to paste.",
    )
    left: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="The left coordinate."
    )
    top: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="The top coordinate."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.image.Paste"


class Resize(GraphNode):
    """
    Change image dimensions to specified width and height.
    image, resize

    - Preprocess images for machine learning model inputs
    - Optimize images for faster web page loading
    - Create uniform image sizes for layouts
    """

    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The image to resize.",
    )
    width: int | GraphNode | tuple[GraphNode, str] = Field(
        default=512, description="The target width."
    )
    height: int | GraphNode | tuple[GraphNode, str] = Field(
        default=512, description="The target height."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.image.Resize"


class SaveImage(GraphNode):
    """
    Save an image to specified asset folder with customizable name format.
    save, image, folder, naming

    Use cases:
    - Save generated images with timestamps
    - Organize outputs into specific folders
    - Create backups of processed images
    """

    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The image to save.",
    )
    folder: types.FolderRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FolderRef(type="folder", uri="", asset_id=None, data=None),
        description="The asset folder to save the image in.",
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="%Y-%m-%d_%H-%M-%S.png",
        description="\n        Name of the output file.\n        You can use time and date variables to create unique names:\n        %Y - Year\n        %m - Month\n        %d - Day\n        %H - Hour\n        %M - Minute\n        %S - Second\n        ",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.image.SaveImage"


class SaveImageFile(GraphNode):
    """
    Write an image to disk.
    image, output, save, file

    Use cases:
    - Save processed images
    - Export edited photos
    - Archive image results
    """

    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The image to save",
    )
    folder: types.FolderPath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FolderPath(type="folder_path", path=""),
        description="Folder where the file will be saved",
    )
    filename: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="\n        The name of the image file.\n        You can use time and date variables to create unique names:\n        %Y - Year\n        %m - Month\n        %d - Day\n        %H - Hour\n        %M - Minute\n        %S - Second\n        ",
    )
    overwrite: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="Overwrite the file if it already exists, otherwise file will be renamed",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.image.SaveImageFile"


class Scale(GraphNode):
    """
    Enlarge or shrink an image by a scale factor.
    image, resize, scale

    - Adjust image dimensions for display galleries
    - Standardize image sizes for machine learning datasets
    - Create thumbnail versions of images
    """

    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The image to scale.",
    )
    scale: float | GraphNode | tuple[GraphNode, str] = Field(
        default=1.0, description="The scale factor."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.image.Scale"
