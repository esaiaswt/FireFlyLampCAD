"""STL export module for funnel STL generator.

Provides binary STL export functionality and data models for export results.
"""

import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
from stl import mesh as stl_mesh, Mode

from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ExportResult:
    """Result of STL export operation."""

    success: bool
    stl_bytes: Optional[bytes]
    face_count: int
    error_message: Optional[str]


def export_binary_stl(
    mesh_obj: stl_mesh.Mesh, filename: str = "funnel_model.stl"
) -> ExportResult:
    """
    Export a numpy-stl Mesh object to binary STL format.

    Updates normal vectors to ensure they are unit-length before export.
    Returns an ExportResult containing the binary STL bytes, face count,
    and success/error status.

    Args:
        mesh_obj: A numpy-stl Mesh object to export.
        filename: The filename embedded in the STL header (default: "funnel_model.stl").

    Returns:
        ExportResult with success=True and stl_bytes on success,
        or success=False and error_message on failure.
    """
    logger.info("Starting binary STL export for filename='%s'", filename)

    try:
        # Update normals to ensure unit-length vectors
        mesh_obj.update_normals()
        logger.debug("Normal vectors updated to unit-length")

        # Get face count
        face_count = len(mesh_obj.data)
        logger.debug("Mesh contains %d faces", face_count)

        # Export to binary STL in memory
        buffer = io.BytesIO()
        mesh_obj.save(filename, fh=buffer, mode=Mode.BINARY)
        stl_bytes = buffer.getvalue()

        file_size = len(stl_bytes)
        logger.info(
            "STL export successful: face_count=%d, file_size=%d bytes",
            face_count,
            file_size,
        )

        return ExportResult(
            success=True,
            stl_bytes=stl_bytes,
            face_count=face_count,
            error_message=None,
        )

    except Exception as e:
        logger.error("STL export failed: %s", e, exc_info=True)
        return ExportResult(
            success=False,
            stl_bytes=None,
            face_count=0,
            error_message=str(e),
        )


def validate_export_roundtrip(
    original: stl_mesh.Mesh, stl_bytes: bytes, tolerance: float = 1e-4
) -> bool:
    """
    Re-import exported STL bytes and compare against the original mesh.

    Validates that the exported STL faithfully represents the original geometry by:
    1. Checking that the face count is identical.
    2. Checking that all vertex positions match within the specified tolerance.

    Args:
        original: The original numpy-stl Mesh object that was exported.
        stl_bytes: The binary STL bytes produced by export_binary_stl().
        tolerance: Maximum allowed positional deviation per coordinate in mm
                   (default: 1e-4 mm).

    Returns:
        True if face count matches and all vertex positions are within tolerance,
        False otherwise.
    """
    logger.debug(
        "Starting export round-trip validation (tolerance=%.2e mm)", tolerance
    )

    try:
        # Re-import STL from bytes
        buffer = io.BytesIO(stl_bytes)
        reimported = stl_mesh.Mesh.from_file("", fh=buffer)

        # Check face count equality
        original_face_count = len(original.data)
        reimported_face_count = len(reimported.data)

        if original_face_count != reimported_face_count:
            logger.debug(
                "Round-trip validation FAILED: face count mismatch "
                "(original=%d, reimported=%d)",
                original_face_count,
                reimported_face_count,
            )
            return False

        logger.debug("Face count matches: %d faces", original_face_count)

        # Check vertex positions within tolerance
        # Each face has 3 vertices (vectors), each vertex has 3 coordinates
        original_vertices = original.vectors  # shape: (N, 3, 3)
        reimported_vertices = reimported.vectors  # shape: (N, 3, 3)

        max_deviation = np.max(np.abs(original_vertices - reimported_vertices))

        if max_deviation > tolerance:
            logger.debug(
                "Round-trip validation FAILED: max vertex deviation %.6e mm "
                "exceeds tolerance %.2e mm",
                max_deviation,
                tolerance,
            )
            return False

        logger.debug(
            "Round-trip validation PASSED: max vertex deviation %.6e mm "
            "(within tolerance %.2e mm)",
            max_deviation,
            tolerance,
        )
        return True

    except Exception as e:
        logger.error("Round-trip validation error: %s", e, exc_info=True)
        return False
