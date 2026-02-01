"""
3D Model Processing Nodes
=========================

This module provides nodes for loading, saving, and processing 3D models
using the Model3DRef type from nodetool-core.

Supports common 3D formats: GLB, GLTF, OBJ, STL, PLY, USDZ.
Includes AI-powered 3D generation nodes (TextTo3D, ImageTo3D).
"""

from __future__ import annotations

import datetime
import os
from enum import Enum
from typing import Any, ClassVar, TypedDict

from pydantic import Field

from nodetool.config.environment import Environment
from nodetool.metadata.types import FolderRef, ImageRef, Model3DModel, Model3DRef, Provider
from nodetool.providers.types import ImageTo3DParams, TextTo3DParams
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext, create_file_uri
from nodetool.workflows.types import SaveUpdate
import aiofiles


# Supported 3D file formats
SUPPORTED_FORMATS = {
    ".glb": "glb",
    ".gltf": "gltf",
    ".obj": "obj",
    ".stl": "stl",
    ".ply": "ply",
    ".usdz": "usdz",
    ".fbx": "fbx",
    ".off": "off",
    ".dae": "dae",
}


class OutputFormat(str, Enum):
    """Supported output formats for 3D model conversion."""

    GLB = "glb"
    GLTF = "gltf"
    OBJ = "obj"
    STL = "stl"
    PLY = "ply"


class BooleanOperation(str, Enum):
    """Boolean operations for combining 3D meshes."""

    UNION = "union"
    DIFFERENCE = "difference"
    INTERSECTION = "intersection"


class ShadingMode(str, Enum):
    """Shading mode for normals calculation."""

    SMOOTH = "smooth"
    FLAT = "flat"
    AUTO = "auto"


def _load_mesh(data: bytes, file_type: str | None = None):
    """Load a mesh from bytes using trimesh."""
    import io

    import trimesh

    # Use file_type hint if provided
    return trimesh.load(io.BytesIO(data), file_type=file_type)


def _export_mesh(mesh, format: str) -> bytes:
    """Export a mesh to bytes in the specified format."""
    import io

    buffer = io.BytesIO()
    mesh.export(buffer, file_type=format)
    buffer.seek(0)
    return buffer.read()


class LoadModel3DFile(BaseNode):
    """
    Load a 3D model file from disk.
    3d, mesh, model, input, load, file, obj, glb, stl, ply

    Use cases:
    - Load 3D models for processing
    - Import meshes from CAD software
    - Read 3D assets for a workflow
    """

    path: str = Field(default="", description="Path to the 3D model file to read")

    async def process(self, context: ProcessingContext) -> Model3DRef:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.path:
            raise ValueError("path cannot be empty")

        expanded_path = os.path.expanduser(self.path)
        if not os.path.exists(expanded_path):
            raise ValueError(f"3D model file not found: {expanded_path}")

        # Get file extension to determine format
        _, ext = os.path.splitext(expanded_path)
        ext_lower = ext.lower()

        if ext_lower not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported formats: {', '.join(SUPPORTED_FORMATS.keys())}"
            )

        async with aiofiles.open(expanded_path, "rb") as f:
            model_data = await f.read()

        # IMPORTANT:
        # Returning large `data` blobs will be serialized into websocket updates and can
        # crash the browser tab. Instead, materialize the model as an asset so the UI
        # can stream it via URL (and downstream nodes can still fetch bytes via asset_id).
        filename = os.path.basename(expanded_path)
        model_format = SUPPORTED_FORMATS[ext_lower]
        result = await context.model3d_from_bytes(
            model_data,
            name=filename,
            format=model_format,
            metadata={"source_uri": create_file_uri(expanded_path)},
        )
        return result


class SaveModel3DFile(BaseNode):
    """
    Save a 3D model to disk.
    3d, mesh, model, output, save, file, export

    Use cases:
    - Save processed 3D models
    - Export meshes to different formats
    - Archive 3D model results
    """

    model: Model3DRef = Field(default=Model3DRef(), description="The 3D model to save")
    folder: str = Field(default="", description="Folder where the file will be saved")
    filename: str = Field(
        default="",
        description="""
        The name of the 3D model file.
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

    async def process(self, context: ProcessingContext) -> Model3DRef:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.folder:
            raise ValueError("folder cannot be empty")
        if not self.filename:
            raise ValueError("filename cannot be empty")
        if self.model.is_empty():
            raise ValueError("The input model is not connected.")

        expanded_folder = os.path.expanduser(self.folder)
        if not os.path.exists(expanded_folder):
            raise ValueError(f"Folder does not exist: {expanded_folder}")

        filename = datetime.datetime.now().strftime(self.filename)
        expanded_path = os.path.join(expanded_folder, filename)
        os.makedirs(os.path.dirname(expanded_path) or ".", exist_ok=True)

        # Get the model data
        if self.model.data:
            model_data = self.model.data
        else:
            model_data = await context.asset_to_bytes(self.model)

        if not self.overwrite:
            count = 1
            while os.path.exists(expanded_path):
                fname, ext = os.path.splitext(filename)
                filename = f"{fname}_{count}{ext}"
                expanded_path = os.path.join(expanded_folder, filename)
                count += 1

        async with aiofiles.open(expanded_path, "wb") as f:
            await f.write(model_data)

        # Determine format from filename
        _, ext = os.path.splitext(expanded_path)
        ext_lower = ext.lower()
        model_format = SUPPORTED_FORMATS.get(ext_lower, self.model.format)

        result = Model3DRef(
            uri=create_file_uri(expanded_path),
            data=model_data,
            format=model_format,
        )

        # Emit SaveUpdate event
        context.post_message(
            SaveUpdate(
                node_id=self.id,
                name=filename,
                value=result,
                output_type="model_3d",
            )
        )

        return result


class SaveModel3D(BaseNode):
    """
    Save a 3D model to an asset folder with customizable name format.
    save, 3d, mesh, model, folder, naming, asset

    Use cases:
    - Save generated 3D models with timestamps
    - Organize outputs into specific folders
    - Create backups of processed models
    """

    model: Model3DRef = Field(default=Model3DRef(), description="The 3D model to save.")
    folder: FolderRef = Field(
        default=FolderRef(),
        description="The asset folder to save the 3D model in.",
    )
    name: str = Field(
        default="%Y-%m-%d_%H-%M-%S.glb",
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
        return "Save Model3D Asset"

    def required_inputs(self):
        return ["model"]

    async def process(self, context: ProcessingContext) -> Model3DRef:
        if self.model.is_empty():
            raise ValueError("The input model is not connected.")

        # Get the model data
        if self.model.data:
            model_data = self.model.data
        else:
            model_data = await context.asset_to_bytes(self.model)

        filename = datetime.datetime.now().strftime(self.name)
        parent_id = self.folder.asset_id if self.folder.is_set() else None

        # Determine format from filename
        _, ext = os.path.splitext(filename)
        ext_lower = ext.lower()
        model_format = SUPPORTED_FORMATS.get(ext_lower, self.model.format)

        # Create asset
        asset = await context.create_asset(
            name=filename,
            content_type=f"model/{model_format or 'gltf-binary'}",
            data=model_data,
            parent_id=parent_id,
        )

        result = Model3DRef(
            uri=await context.get_asset_url(asset.id),
            asset_id=asset.id,
            format=model_format,
        )

        # Emit SaveUpdate event
        context.post_message(
            SaveUpdate(
                node_id=self.id,
                name=filename,
                value=result,
                output_type="model_3d",
            )
        )

        return result

    def result_for_client(self, result: dict[str, Any]) -> dict[str, Any]:
        return self.result_for_all_outputs(result)


class FormatConverter(BaseNode):
    """
    Convert a 3D model to a different format.
    3d, mesh, model, convert, format, obj, glb, stl, ply, usdz, export

    Use cases:
    - Convert high-poly sculpts to web-friendly GLB
    - Export models for 3D printing (STL)
    - Create cross-platform 3D assets
    """

    model: Model3DRef = Field(
        default=Model3DRef(), description="The 3D model to convert"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.GLB,
        description="Target format for conversion",
    )

    async def process(self, context: ProcessingContext) -> Model3DRef:
        if self.model.is_empty():
            raise ValueError("The input model is not connected.")

        # Get the model data
        if self.model.data:
            model_data = self.model.data
        else:
            model_data = await context.asset_to_bytes(self.model)

        # Load and convert the mesh
        mesh = _load_mesh(model_data, file_type=self.model.format)
        converted_data = _export_mesh(mesh, self.output_format.value)

        return Model3DRef(
            data=converted_data,
            format=self.output_format.value,
        )


class GetModel3DMetadata(BaseNode):
    """
    Get metadata about a 3D model.
    3d, mesh, model, metadata, info, properties

    Use cases:
    - Get vertex and face counts for processing decisions
    - Analyze model properties
    - Gather information for model cataloging
    """

    model: Model3DRef = Field(
        default=Model3DRef(), description="The 3D model to analyze"
    )

    class OutputType(TypedDict):
        format: str
        vertex_count: int
        face_count: int
        is_watertight: bool
        bounds_min: list[float]
        bounds_max: list[float]
        center_of_mass: list[float]
        volume: float
        surface_area: float

    async def process(self, context: ProcessingContext) -> OutputType:
        if self.model.is_empty():
            raise ValueError("The input model is not connected.")

        import trimesh

        # Get the model data
        if self.model.data:
            model_data = self.model.data
        else:
            model_data = await context.asset_to_bytes(self.model)

        # Load the mesh
        mesh = _load_mesh(model_data, file_type=self.model.format)

        # Handle scene vs mesh
        if isinstance(mesh, trimesh.Scene):
            # Combine all meshes in the scene
            mesh = mesh.dump(concatenate=True)

        return {
            "format": self.model.format or "unknown",
            "vertex_count": len(mesh.vertices),
            "face_count": len(mesh.faces),
            "is_watertight": mesh.is_watertight,
            "bounds_min": mesh.bounds[0].tolist(),
            "bounds_max": mesh.bounds[1].tolist(),
            "center_of_mass": mesh.center_mass.tolist(),
            "volume": float(mesh.volume) if mesh.is_watertight else 0.0,
            "surface_area": float(mesh.area),
        }


class Transform3D(BaseNode):
    """
    Apply translation, rotation, and scaling to a 3D model.
    3d, mesh, model, transform, translate, rotate, scale, move

    Use cases:
    - Position models in 3D space
    - Scale models to specific dimensions
    - Rotate models for proper orientation
    """

    model: Model3DRef = Field(
        default=Model3DRef(), description="The 3D model to transform"
    )
    translate_x: float = Field(default=0.0, description="Translation along X axis")
    translate_y: float = Field(default=0.0, description="Translation along Y axis")
    translate_z: float = Field(default=0.0, description="Translation along Z axis")
    rotate_x: float = Field(
        default=0.0, ge=-360, le=360, description="Rotation around X axis in degrees"
    )
    rotate_y: float = Field(
        default=0.0, ge=-360, le=360, description="Rotation around Y axis in degrees"
    )
    rotate_z: float = Field(
        default=0.0, ge=-360, le=360, description="Rotation around Z axis in degrees"
    )
    scale_x: float = Field(default=1.0, gt=0, description="Scale factor along X axis")
    scale_y: float = Field(default=1.0, gt=0, description="Scale factor along Y axis")
    scale_z: float = Field(default=1.0, gt=0, description="Scale factor along Z axis")
    uniform_scale: float = Field(
        default=1.0, gt=0, description="Uniform scale factor (applied after axis scales)"
    )

    async def process(self, context: ProcessingContext) -> Model3DRef:
        if self.model.is_empty():
            raise ValueError("The input model is not connected.")

        import numpy as np
        import trimesh

        # Get the model data
        if self.model.data:
            model_data = self.model.data
        else:
            model_data = await context.asset_to_bytes(self.model)

        # Load the mesh
        mesh = _load_mesh(model_data, file_type=self.model.format)

        # Handle scene vs mesh
        is_scene = isinstance(mesh, trimesh.Scene)
        if is_scene:
            mesh = mesh.dump(concatenate=True)

        # Build transformation matrix
        # Scale
        scale_matrix = np.diag(
            [
                self.scale_x * self.uniform_scale,
                self.scale_y * self.uniform_scale,
                self.scale_z * self.uniform_scale,
                1.0,
            ]
        )

        # Rotation (convert degrees to radians)
        rx = np.radians(self.rotate_x)
        ry = np.radians(self.rotate_y)
        rz = np.radians(self.rotate_z)

        # Rotation matrices
        rot_x = trimesh.transformations.rotation_matrix(rx, [1, 0, 0])
        rot_y = trimesh.transformations.rotation_matrix(ry, [0, 1, 0])
        rot_z = trimesh.transformations.rotation_matrix(rz, [0, 0, 1])

        # Translation
        translation = trimesh.transformations.translation_matrix(
            [self.translate_x, self.translate_y, self.translate_z]
        )

        # Combine transformations: scale -> rotate -> translate
        transform = translation @ rot_z @ rot_y @ rot_x @ scale_matrix
        mesh.apply_transform(transform)

        # Export
        output_format = self.model.format or "glb"
        converted_data = _export_mesh(mesh, output_format)

        return Model3DRef(
            data=converted_data,
            format=output_format,
        )


class Decimate(BaseNode):
    """
    Reduce polygon count while preserving shape using Quadric Error Metrics.
    3d, mesh, model, decimate, simplify, reduce, polygon, optimize, LOD

    Use cases:
    - Create level-of-detail (LOD) versions
    - Optimize models for real-time rendering
    - Reduce file size for web deployment
    - Prepare models for mobile/VR applications
    """

    model: Model3DRef = Field(
        default=Model3DRef(), description="The 3D model to decimate"
    )
    target_ratio: float = Field(
        default=0.5,
        ge=0.01,
        le=1.0,
        description="Target ratio of faces to keep (0.5 = 50% reduction)",
    )

    async def process(self, context: ProcessingContext) -> Model3DRef:
        if self.model.is_empty():
            raise ValueError("The input model is not connected.")

        import trimesh

        # Get the model data
        if self.model.data:
            model_data = self.model.data
        else:
            model_data = await context.asset_to_bytes(self.model)

        # Load the mesh
        mesh = _load_mesh(model_data, file_type=self.model.format)

        # Handle scene vs mesh
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        # Calculate target face count
        original_faces = len(mesh.faces)
        target_faces = max(4, int(original_faces * self.target_ratio))

        # Simplify using quadric decimation
        simplified = mesh.simplify_quadric_decimation(face_count=target_faces)

        # Export
        output_format = self.model.format or "glb"
        converted_data = _export_mesh(simplified, output_format)

        return Model3DRef(
            data=converted_data,
            format=output_format,
            metadata={
                "original_faces": original_faces,
                "simplified_faces": len(simplified.faces),
                "reduction_ratio": 1.0 - (len(simplified.faces) / original_faces),
            },
        )


class Boolean3D(BaseNode):
    """
    Perform boolean operations on 3D meshes.
    3d, mesh, model, boolean, union, difference, intersection, combine, subtract

    Use cases:
    - Combine multiple objects (union)
    - Cut holes in objects (difference)
    - Find overlapping regions (intersection)
    - Hard-surface modeling operations
    - 3D printing preparation
    """

    model_a: Model3DRef = Field(
        default=Model3DRef(), description="First 3D model (base)"
    )
    model_b: Model3DRef = Field(
        default=Model3DRef(), description="Second 3D model (tool)"
    )
    operation: BooleanOperation = Field(
        default=BooleanOperation.UNION,
        description="Boolean operation to perform",
    )

    async def process(self, context: ProcessingContext) -> Model3DRef:
        if self.model_a.is_empty():
            raise ValueError("Model A is not connected.")
        if self.model_b.is_empty():
            raise ValueError("Model B is not connected.")

        import trimesh

        # Get model data
        if self.model_a.data:
            data_a = self.model_a.data
        else:
            data_a = await context.asset_to_bytes(self.model_a)

        if self.model_b.data:
            data_b = self.model_b.data
        else:
            data_b = await context.asset_to_bytes(self.model_b)

        # Load meshes
        mesh_a = _load_mesh(data_a, file_type=self.model_a.format)
        mesh_b = _load_mesh(data_b, file_type=self.model_b.format)

        # Handle scenes
        if isinstance(mesh_a, trimesh.Scene):
            mesh_a = mesh_a.dump(concatenate=True)
        if isinstance(mesh_b, trimesh.Scene):
            mesh_b = mesh_b.dump(concatenate=True)

        # Perform boolean operation
        if self.operation == BooleanOperation.UNION:
            result = mesh_a.union(mesh_b, engine="blender")
        elif self.operation == BooleanOperation.DIFFERENCE:
            result = mesh_a.difference(mesh_b, engine="blender")
        elif self.operation == BooleanOperation.INTERSECTION:
            result = mesh_a.intersection(mesh_b, engine="blender")
        else:
            raise ValueError(f"Unknown operation: {self.operation}")

        # Export
        output_format = self.model_a.format or "glb"
        converted_data = _export_mesh(result, output_format)

        return Model3DRef(
            data=converted_data,
            format=output_format,
        )


class RecalculateNormals(BaseNode):
    """
    Recalculate mesh normals for proper shading.
    3d, mesh, model, normals, fix, shading, smooth, flat, faces

    Use cases:
    - Fix inverted or broken normals
    - Switch between smooth and flat shading
    - Repair imported meshes with bad normals
    - Prepare models for rendering
    """

    model: Model3DRef = Field(
        default=Model3DRef(), description="The 3D model to process"
    )
    mode: ShadingMode = Field(
        default=ShadingMode.AUTO,
        description="Shading mode: smooth, flat, or auto (uses mesh default)",
    )
    fix_winding: bool = Field(
        default=True,
        description="Fix inconsistent face winding (inverted faces)",
    )

    async def process(self, context: ProcessingContext) -> Model3DRef:
        if self.model.is_empty():
            raise ValueError("The input model is not connected.")

        import trimesh

        # Get the model data
        if self.model.data:
            model_data = self.model.data
        else:
            model_data = await context.asset_to_bytes(self.model)

        # Load the mesh
        mesh = _load_mesh(model_data, file_type=self.model.format)

        # Handle scene vs mesh
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        # Fix winding if requested
        if self.fix_winding:
            trimesh.repair.fix_winding(mesh)
            trimesh.repair.fix_normals(mesh)

        # Recalculate normals based on mode
        if self.mode == ShadingMode.FLAT:
            # For flat shading, unmerge vertices so each face has unique vertices
            # This ensures each face can have its own normal
            mesh.unmerge_vertices()
            # Recompute face normals
            mesh.face_normals
        elif self.mode == ShadingMode.SMOOTH:
            # Compute smooth vertex normals (area-weighted average of face normals)
            mesh.vertex_normals
        else:
            # Auto mode: let trimesh handle it with default behavior
            # This uses area-weighted vertex normals
            mesh.vertex_normals

        # Export
        output_format = self.model.format or "glb"
        converted_data = _export_mesh(mesh, output_format)

        return Model3DRef(
            data=converted_data,
            format=output_format,
        )


class CenterMesh(BaseNode):
    """
    Center a mesh at the origin.
    3d, mesh, model, center, origin, align

    Use cases:
    - Center models for consistent positioning
    - Prepare models for rotation
    - Align multiple models
    """

    model: Model3DRef = Field(
        default=Model3DRef(), description="The 3D model to center"
    )
    use_centroid: bool = Field(
        default=True,
        description="Use geometric centroid (True) or bounding box center (False)",
    )

    async def process(self, context: ProcessingContext) -> Model3DRef:
        if self.model.is_empty():
            raise ValueError("The input model is not connected.")

        import trimesh

        # Get the model data
        if self.model.data:
            model_data = self.model.data
        else:
            model_data = await context.asset_to_bytes(self.model)

        # Load the mesh
        mesh = _load_mesh(model_data, file_type=self.model.format)

        # Handle scene vs mesh
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        # Center the mesh
        if self.use_centroid:
            center = mesh.centroid
        else:
            center = mesh.bounding_box.centroid

        mesh.vertices -= center

        # Export
        output_format = self.model.format or "glb"
        converted_data = _export_mesh(mesh, output_format)

        return Model3DRef(
            data=converted_data,
            format=output_format,
        )


class FlipNormals(BaseNode):
    """
    Flip all face normals of a mesh.
    3d, mesh, model, normals, flip, invert, inside_out

    Use cases:
    - Fix inside-out meshes
    - Invert normals for specific rendering effects
    - Repair meshes from incompatible software
    """

    model: Model3DRef = Field(
        default=Model3DRef(), description="The 3D model to process"
    )

    async def process(self, context: ProcessingContext) -> Model3DRef:
        if self.model.is_empty():
            raise ValueError("The input model is not connected.")

        import trimesh

        # Get the model data
        if self.model.data:
            model_data = self.model.data
        else:
            model_data = await context.asset_to_bytes(self.model)

        # Load the mesh
        mesh = _load_mesh(model_data, file_type=self.model.format)

        # Handle scene vs mesh
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        # Flip normals by reversing face winding
        mesh.invert()

        # Export
        output_format = self.model.format or "glb"
        converted_data = _export_mesh(mesh, output_format)

        return Model3DRef(
            data=converted_data,
            format=output_format,
        )


class MergeMeshes(BaseNode):
    """
    Merge multiple meshes into a single mesh.
    3d, mesh, model, merge, combine, concatenate

    Use cases:
    - Combine multiple parts into one model
    - Merge imported components
    - Prepare models for boolean operations
    """

    models: list[Model3DRef] = Field(
        default=[],
        description="List of 3D models to merge",
    )

    async def process(self, context: ProcessingContext) -> Model3DRef:
        if not self.models:
            raise ValueError("No models provided to merge.")

        import trimesh

        meshes = []
        output_format = None

        for model in self.models:
            if model.is_empty():
                continue

            # Get model data
            if model.data:
                data = model.data
            else:
                data = await context.asset_to_bytes(model)

            # Load mesh
            mesh = _load_mesh(data, file_type=model.format)

            # Handle scene vs mesh
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)

            meshes.append(mesh)

            # Use format from first model
            if output_format is None:
                output_format = model.format

        if not meshes:
            raise ValueError("No valid meshes to merge.")

        # Concatenate all meshes
        merged = trimesh.util.concatenate(meshes)

        # Export
        output_format = output_format or "glb"
        converted_data = _export_mesh(merged, output_format)

        return Model3DRef(
            data=converted_data,
            format=output_format,
            metadata={
                "mesh_count": len(meshes),
            },
        )




class TextTo3D(BaseNode):
    """
    Generate 3D models from text prompts using AI providers (Meshy, Rodin).
    3d, generation, AI, text-to-3d, t3d, mesh, create

    Use cases:
    - Create 3D models from text descriptions
    - Generate game assets from prompts
    - Prototype 3D concepts quickly
    - Create 3D content for AR/VR
    """

    _auto_save_asset: ClassVar[bool] = True
    _expose_as_tool: ClassVar[bool] = True

    model: Model3DModel = Field(
        default=Model3DModel(
            provider=Provider.Meshy,
            id="meshy-4",
            name="Meshy-4 Text-to-3D",
        ),
        description="The 3D generation model to use",
    )
    prompt: str = Field(
        default="",
        description="Text description of the 3D model to generate",
    )
    negative_prompt: str = Field(
        default="",
        description="Elements to avoid in the generated model",
    )
    art_style: str = Field(
        default="",
        description="Art style for the model (e.g., 'realistic', 'cartoon', 'low-poly')",
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.GLB,
        description="Output format for the 3D model",
    )
    seed: int = Field(
        default=-1,
        ge=-1,
        description="Random seed for reproducibility (-1 for random)",
    )
    timeout_seconds: int = Field(
        default=600,
        ge=0,
        le=7200,
        description="Timeout in seconds for API calls (0 = use provider default)",
    )

    async def process(self, context: ProcessingContext) -> Model3DRef:
        if not self.prompt:
            raise ValueError("Prompt is required for text-to-3D generation")

        # Get the 3D provider for this model
        provider_instance = await context.get_provider(self.model.provider)

        params = TextTo3DParams(
            model=self.model,
            prompt=self.prompt,
            negative_prompt=self.negative_prompt if self.negative_prompt else None,
            art_style=self.art_style if self.art_style else None,
            output_format=self.output_format.value,
            seed=self.seed if self.seed != -1 else None,
        )

        # Generate 3D model
        model_bytes = await provider_instance.text_to_3d(
            params,
            timeout_s=self.timeout_seconds if self.timeout_seconds > 0 else None,
            context=context,
            node_id=self.id,
        )

        # Convert to Model3DRef (creates asset to avoid large websocket payloads)
        return await context.model3d_from_bytes(
            model_bytes,
            name=f"generated_{self.id}.{self.output_format.value}",
            format=self.output_format.value,
        )

    @classmethod
    def get_basic_fields(cls):
        return ["model", "prompt", "output_format", "seed"]


class ImageTo3D(BaseNode):
    """
    Generate 3D models from images using AI providers (Meshy, Rodin).
    3d, generation, AI, image-to-3d, i3d, mesh, reconstruction

    Use cases:
    - Convert product photos to 3D models
    - Create 3D assets from concept art
    - Generate 3D characters from drawings
    - Reconstruct objects from images
    """

    _auto_save_asset: ClassVar[bool] = True
    _expose_as_tool: ClassVar[bool] = True

    model: Model3DModel = Field(
        default=Model3DModel(
            provider=Provider.Meshy,
            id="meshy-4-image",
            name="Meshy-4 Image-to-3D",
        ),
        description="The 3D generation model to use",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        description="Input image to convert to 3D",
    )
    prompt: str = Field(
        default="",
        description="Optional text prompt to guide the 3D generation",
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.GLB,
        description="Output format for the 3D model",
    )
    seed: int = Field(
        default=-1,
        ge=-1,
        description="Random seed for reproducibility (-1 for random)",
    )
    timeout_seconds: int = Field(
        default=600,
        ge=0,
        le=7200,
        description="Timeout in seconds for API calls (0 = use provider default)",
    )

    async def process(self, context: ProcessingContext) -> Model3DRef:
        if self.image.is_empty():
            raise ValueError("Input image must be connected")

        # Get the 3D provider for this model
        provider_instance = await context.get_provider(self.model.provider)

        # Read the image bytes from the ImageRef
        image_io = await context.asset_to_io(self.image)
        image_bytes = image_io.read()

        params = ImageTo3DParams(
            model=self.model,
            prompt=self.prompt if self.prompt else None,
            output_format=self.output_format.value,
            seed=self.seed if self.seed != -1 else None,
        )

        # Generate 3D model
        model_bytes = await provider_instance.image_to_3d(
            image_bytes,
            params,
            timeout_s=self.timeout_seconds if self.timeout_seconds > 0 else None,
            context=context,
            node_id=self.id,
        )

        # Convert to Model3DRef (creates asset to avoid large websocket payloads)
        return await context.model3d_from_bytes(
            model_bytes,
            name=f"generated_{self.id}.{self.output_format.value}",
            format=self.output_format.value,
        )

    @classmethod
    def get_basic_fields(cls):
        return ["model", "image", "output_format", "seed"]


__all__ = [
    "LoadModel3DFile",
    "SaveModel3DFile",
    "SaveModel3D",
    "FormatConverter",
    "GetModel3DMetadata",
    "Transform3D",
    "Decimate",
    "Boolean3D",
    "RecalculateNormals",
    "CenterMesh",
    "FlipNormals",
    "MergeMeshes",
    "TextTo3D",
    "ImageTo3D",
]
