"""Mesh validation module for funnel STL generator.

Provides mesh quality validation using trimesh for analysis.
Checks watertight status, non-manifold edges, degenerate faces,
and normal consistency.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import trimesh
from stl import mesh as stl_mesh

from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MeshValidationResult:
    """Result of mesh quality validation."""

    is_watertight: bool
    non_manifold_edge_count: int
    degenerate_face_count: int
    normals_consistent: bool
    errors: List[str] = field(default_factory=list)


def validate_mesh(mesh_obj: stl_mesh.Mesh) -> MeshValidationResult:
    """
    Validate mesh quality by converting to trimesh and running checks.

    Performs the following validations:
    - Watertight (manifold) check
    - Non-manifold edge detection
    - Degenerate face detection (area < 1e-6 mm²)
    - Normal consistency check (positive volume indicates consistent outward normals)

    Args:
        mesh_obj: A numpy-stl Mesh object to validate.

    Returns:
        MeshValidationResult with all validation findings.
    """
    logger.info("validate_mesh() entry - starting mesh validation")

    # Convert numpy-stl mesh to trimesh
    # Each face in numpy-stl has 3 vertices (triangle), flatten and re-index
    vectors = mesh_obj.vectors.reshape(-1, 3)
    faces = np.arange(len(vectors)).reshape(-1, 3)
    tri_mesh = trimesh.Trimesh(vertices=vectors, faces=faces, process=True)

    errors: List[str] = []

    # Check watertight (manifold) status
    is_watertight = bool(tri_mesh.is_watertight)
    if not is_watertight:
        errors.append("Mesh is not watertight (not manifold)")

    # Count non-manifold edges
    # In trimesh, edges that appear in more than 2 faces or only 1 face are non-manifold
    # face_adjacency gives pairs of faces sharing an edge; edges not shared by exactly 2 faces are non-manifold
    edges = tri_mesh.edges_sorted
    unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
    # Non-manifold edges are those shared by != 2 faces
    non_manifold_edge_count = int(np.sum(edge_counts != 2))
    if non_manifold_edge_count > 0:
        errors.append(
            f"Mesh has {non_manifold_edge_count} non-manifold edge(s)"
        )

    # Detect degenerate faces (area < 1e-6 mm²)
    face_areas = tri_mesh.area_faces
    degenerate_face_count = int(np.sum(face_areas < 1e-6))
    if degenerate_face_count > 0:
        errors.append(
            f"Mesh has {degenerate_face_count} degenerate face(s) "
            f"(area < 1e-6 mm²)"
        )

    # Check normal consistency
    # A positive volume indicates consistently oriented outward normals
    # trimesh computes signed volume from face normals; if normals are inconsistent,
    # volume computation will be unreliable or negative
    normals_consistent = bool(tri_mesh.volume > 0) if is_watertight else False
    if not normals_consistent:
        if is_watertight:
            errors.append("Mesh normals are not consistently oriented outward")
        else:
            errors.append(
                "Normal consistency could not be verified (mesh is not watertight)"
            )

    result = MeshValidationResult(
        is_watertight=is_watertight,
        non_manifold_edge_count=non_manifold_edge_count,
        degenerate_face_count=degenerate_face_count,
        normals_consistent=normals_consistent,
        errors=errors,
    )

    logger.info(
        f"validate_mesh() complete - watertight={is_watertight}, "
        f"non_manifold_edges={non_manifold_edge_count}, "
        f"degenerate_faces={degenerate_face_count}, "
        f"normals_consistent={normals_consistent}, "
        f"errors={len(errors)}"
    )

    return result
