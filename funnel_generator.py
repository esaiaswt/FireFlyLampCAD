"""Funnel and cylinder sleeve mesh generation module.

Provides pure functions that produce mesh geometry from FunnelParams.
Uses numpy-stl for mesh creation with NumPy-backed vectorized operations.
"""

import time
import numpy as np
from stl import mesh as stl_mesh

from logging_config import get_logger
from parameter_validator import FunnelParams

logger = get_logger(__name__)


def generate_funnel_mesh(params: FunnelParams) -> stl_mesh.Mesh:
    """
    Generate a hollow truncated cone (funnel) mesh.

    The mesh consists of four surfaces:
    - Outer conical surface (normals point outward)
    - Inner conical surface (normals point toward center/hollow interior)
    - Bottom annular ring (normals point downward, -Z)
    - Top annular ring (normals point upward, +Z)

    Inner diameters are computed as outer - 2 * wall_thickness.
    Uses num_segments points around the circumference with consistent vertex
    winding for correct normal orientation.

    Args:
        params: FunnelParams with funnel dimensions.

    Returns:
        A numpy-stl Mesh object representing the hollow funnel.
    """
    start_time = time.perf_counter()
    logger.debug(
        "generate_funnel_mesh() entry - "
        f"bottom_diameter={params.bottom_diameter}, top_diameter={params.top_diameter}, "
        f"funnel_height={params.funnel_height}, wall_thickness={params.wall_thickness}, "
        f"num_segments={params.num_segments}"
    )

    n = params.num_segments

    # Radii
    r_outer_bottom = params.bottom_diameter / 2.0
    r_outer_top = params.top_diameter / 2.0
    r_inner_bottom = (params.bottom_diameter - 2.0 * params.wall_thickness) / 2.0
    r_inner_top = (params.top_diameter - 2.0 * params.wall_thickness) / 2.0

    height = params.funnel_height

    # Angles for circumferential vertices
    angles = np.linspace(0, 2.0 * np.pi, n, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    # Each surface (outer, inner, top ring, bottom ring) has n quads = 2n triangles
    # Total faces: 4 surfaces * 2n = 8n
    total_faces = 8 * n
    faces = np.zeros(total_faces, dtype=stl_mesh.Mesh.dtype)

    face_idx = 0

    # --- Outer conical surface (normals point outward) ---
    # CCW winding when viewed from outside produces outward normals
    for i in range(n):
        i_next = (i + 1) % n

        # Bottom outer vertices
        bx0 = r_outer_bottom * cos_a[i]
        by0 = r_outer_bottom * sin_a[i]
        bx1 = r_outer_bottom * cos_a[i_next]
        by1 = r_outer_bottom * sin_a[i_next]
        # Top outer vertices
        tx0 = r_outer_top * cos_a[i]
        ty0 = r_outer_top * sin_a[i]
        tx1 = r_outer_top * cos_a[i_next]
        ty1 = r_outer_top * sin_a[i_next]

        # Triangle 1: bottom[i], bottom[i+1], top[i+1]
        faces['vectors'][face_idx] = [
            [bx0, by0, 0.0],
            [bx1, by1, 0.0],
            [tx1, ty1, height]
        ]
        face_idx += 1

        # Triangle 2: bottom[i], top[i+1], top[i]
        faces['vectors'][face_idx] = [
            [bx0, by0, 0.0],
            [tx1, ty1, height],
            [tx0, ty0, height]
        ]
        face_idx += 1

    # --- Inner conical surface (normals point toward center / into hollow) ---
    # Reversed winding compared to outer surface
    for i in range(n):
        i_next = (i + 1) % n

        # Bottom inner vertices
        bx0 = r_inner_bottom * cos_a[i]
        by0 = r_inner_bottom * sin_a[i]
        bx1 = r_inner_bottom * cos_a[i_next]
        by1 = r_inner_bottom * sin_a[i_next]
        # Top inner vertices
        tx0 = r_inner_top * cos_a[i]
        ty0 = r_inner_top * sin_a[i]
        tx1 = r_inner_top * cos_a[i_next]
        ty1 = r_inner_top * sin_a[i_next]

        # Triangle 1: bottom[i], top[i], top[i+1]
        # Reversed winding for inward-facing normals
        faces['vectors'][face_idx] = [
            [bx0, by0, 0.0],
            [tx0, ty0, height],
            [tx1, ty1, height]
        ]
        face_idx += 1

        # Triangle 2: bottom[i], top[i+1], bottom[i+1]
        faces['vectors'][face_idx] = [
            [bx0, by0, 0.0],
            [tx1, ty1, height],
            [bx1, by1, 0.0]
        ]
        face_idx += 1

    # --- Bottom annular ring (z=0, normals point downward -Z) ---
    # Connects outer bottom to inner bottom
    for i in range(n):
        i_next = (i + 1) % n

        # Outer bottom vertices
        ox0 = r_outer_bottom * cos_a[i]
        oy0 = r_outer_bottom * sin_a[i]
        ox1 = r_outer_bottom * cos_a[i_next]
        oy1 = r_outer_bottom * sin_a[i_next]
        # Inner bottom vertices
        ix0 = r_inner_bottom * cos_a[i]
        iy0 = r_inner_bottom * sin_a[i]
        ix1 = r_inner_bottom * cos_a[i_next]
        iy1 = r_inner_bottom * sin_a[i_next]

        # Triangle 1: outer[i], inner[i+1], outer[i+1]
        # CCW when viewed from below (-Z) = downward normal
        faces['vectors'][face_idx] = [
            [ox0, oy0, 0.0],
            [ix1, iy1, 0.0],
            [ox1, oy1, 0.0]
        ]
        face_idx += 1

        # Triangle 2: outer[i], inner[i], inner[i+1]
        faces['vectors'][face_idx] = [
            [ox0, oy0, 0.0],
            [ix0, iy0, 0.0],
            [ix1, iy1, 0.0]
        ]
        face_idx += 1

    # --- Top annular ring (z=height, normals point upward +Z) ---
    # Connects outer top to inner top
    for i in range(n):
        i_next = (i + 1) % n

        # Outer top vertices
        ox0 = r_outer_top * cos_a[i]
        oy0 = r_outer_top * sin_a[i]
        ox1 = r_outer_top * cos_a[i_next]
        oy1 = r_outer_top * sin_a[i_next]
        # Inner top vertices
        ix0 = r_inner_top * cos_a[i]
        iy0 = r_inner_top * sin_a[i]
        ix1 = r_inner_top * cos_a[i_next]
        iy1 = r_inner_top * sin_a[i_next]

        # Triangle 1: outer[i], outer[i+1], inner[i+1]
        # CCW when viewed from above (+Z) = upward normal
        faces['vectors'][face_idx] = [
            [ox0, oy0, height],
            [ox1, oy1, height],
            [ix1, iy1, height]
        ]
        face_idx += 1

        # Triangle 2: outer[i], inner[i+1], inner[i]
        faces['vectors'][face_idx] = [
            [ox0, oy0, height],
            [ix1, iy1, height],
            [ix0, iy0, height]
        ]
        face_idx += 1

    # Create the mesh object
    funnel_mesh = stl_mesh.Mesh(faces)

    # Update normals from vertex winding order
    funnel_mesh.update_normals()

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.debug(
        f"generate_funnel_mesh() exit - "
        f"faces={total_faces}, elapsed={elapsed_ms:.2f}ms"
    )

    return funnel_mesh


def generate_cylinder_sleeve_mesh(params: FunnelParams) -> stl_mesh.Mesh:
    """
    Generate a hollow cylinder sleeve mesh.

    The cylinder sleeve wraps the outer circumference of the funnel top,
    extending downward from the funnel brim.

    Geometry:
    - Inner diameter = params.top_diameter (wraps the funnel top)
    - Outer diameter = params.top_diameter + 2 * params.sleeve_wall_thickness
    - Top edge Z = params.funnel_height (brim alignment)
    - Bottom edge Z = params.funnel_height - params.sleeve_height

    The mesh consists of:
    - Outer cylindrical surface (normals pointing outward)
    - Inner cylindrical surface (normals pointing inward)
    - Top annular ring
    - Bottom annular ring

    Each quad is split into 2 triangles with consistent vertex winding
    for outward-facing normals.

    Args:
        params: FunnelParams defining the cylinder sleeve dimensions.

    Returns:
        A numpy-stl Mesh object representing the cylinder sleeve.
    """
    start_time = time.perf_counter()
    logger.debug(
        "generate_cylinder_sleeve_mesh() entry - "
        f"top_diameter={params.top_diameter}, sleeve_height={params.sleeve_height}, "
        f"sleeve_wall_thickness={params.sleeve_wall_thickness}, "
        f"num_segments={params.num_segments}"
    )

    n = params.num_segments

    # Radii
    inner_radius = params.top_diameter / 2.0
    outer_radius = inner_radius + params.sleeve_wall_thickness

    # Z positions - sleeve extends downward from brim
    z_top = params.funnel_height
    z_bottom = params.funnel_height - params.sleeve_height

    # Angle steps
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    # Each surface (outer, inner, top ring, bottom ring) has n quads = 2n triangles
    # Total faces: 4 surfaces * 2n triangles = 8n
    total_faces = 8 * n
    faces = np.zeros(total_faces, dtype=stl_mesh.Mesh.dtype)

    face_idx = 0

    # --- Outer cylindrical surface (normals point outward) ---
    for i in range(n):
        i_next = (i + 1) % n

        # Vertices on outer surface
        # Bottom vertices
        bx0 = outer_radius * cos_a[i]
        by0 = outer_radius * sin_a[i]
        bx1 = outer_radius * cos_a[i_next]
        by1 = outer_radius * sin_a[i_next]
        # Top vertices
        tx0 = bx0
        ty0 = by0
        tx1 = bx1
        ty1 = by1

        # Triangle 1: bottom-left, bottom-right, top-right
        # Winding: CCW when viewed from outside = outward normal
        faces['vectors'][face_idx] = [
            [bx0, by0, z_bottom],
            [bx1, by1, z_bottom],
            [tx1, ty1, z_top]
        ]
        face_idx += 1

        # Triangle 2: bottom-left, top-right, top-left
        faces['vectors'][face_idx] = [
            [bx0, by0, z_bottom],
            [tx1, ty1, z_top],
            [tx0, ty0, z_top]
        ]
        face_idx += 1

    # --- Inner cylindrical surface (normals point inward, toward center) ---
    for i in range(n):
        i_next = (i + 1) % n

        # Vertices on inner surface
        bx0 = inner_radius * cos_a[i]
        by0 = inner_radius * sin_a[i]
        bx1 = inner_radius * cos_a[i_next]
        by1 = inner_radius * sin_a[i_next]
        tx0 = bx0
        ty0 = by0
        tx1 = bx1
        ty1 = by1

        # For inward normals, reverse winding compared to outer surface
        # Triangle 1: bottom-left, top-right, bottom-right
        faces['vectors'][face_idx] = [
            [bx0, by0, z_bottom],
            [tx0, ty0, z_top],
            [tx1, ty1, z_top]
        ]
        face_idx += 1

        # Triangle 2: bottom-left, top-right-next, bottom-right
        faces['vectors'][face_idx] = [
            [bx0, by0, z_bottom],
            [tx1, ty1, z_top],
            [bx1, by1, z_bottom]
        ]
        face_idx += 1

    # --- Top annular ring (at z_top, normal points up +Z) ---
    for i in range(n):
        i_next = (i + 1) % n

        # Outer vertices at top
        ox0 = outer_radius * cos_a[i]
        oy0 = outer_radius * sin_a[i]
        ox1 = outer_radius * cos_a[i_next]
        oy1 = outer_radius * sin_a[i_next]
        # Inner vertices at top
        ix0 = inner_radius * cos_a[i]
        iy0 = inner_radius * sin_a[i]
        ix1 = inner_radius * cos_a[i_next]
        iy1 = inner_radius * sin_a[i_next]

        # Triangle 1: inner_i, outer_i, outer_next
        # CCW when viewed from above (+Z) = upward normal
        faces['vectors'][face_idx] = [
            [ix0, iy0, z_top],
            [ox0, oy0, z_top],
            [ox1, oy1, z_top]
        ]
        face_idx += 1

        # Triangle 2: inner_i, outer_next, inner_next
        faces['vectors'][face_idx] = [
            [ix0, iy0, z_top],
            [ox1, oy1, z_top],
            [ix1, iy1, z_top]
        ]
        face_idx += 1

    # --- Bottom annular ring (at z_bottom, normal points down -Z) ---
    for i in range(n):
        i_next = (i + 1) % n

        # Outer vertices at bottom
        ox0 = outer_radius * cos_a[i]
        oy0 = outer_radius * sin_a[i]
        ox1 = outer_radius * cos_a[i_next]
        oy1 = outer_radius * sin_a[i_next]
        # Inner vertices at bottom
        ix0 = inner_radius * cos_a[i]
        iy0 = inner_radius * sin_a[i]
        ix1 = inner_radius * cos_a[i_next]
        iy1 = inner_radius * sin_a[i_next]

        # Triangle 1: inner_i, outer_next, outer_i
        # CCW when viewed from below (-Z) = downward normal
        faces['vectors'][face_idx] = [
            [ix0, iy0, z_bottom],
            [ox1, oy1, z_bottom],
            [ox0, oy0, z_bottom]
        ]
        face_idx += 1

        # Triangle 2: inner_i, inner_next, outer_next
        faces['vectors'][face_idx] = [
            [ix0, iy0, z_bottom],
            [ix1, iy1, z_bottom],
            [ox1, oy1, z_bottom]
        ]
        face_idx += 1

    # Create the mesh object
    cylinder_mesh = stl_mesh.Mesh(faces)

    # Update normals to match the vertex winding
    cylinder_mesh.update_normals()

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.debug(
        f"generate_cylinder_sleeve_mesh() exit - "
        f"faces={total_faces}, elapsed={elapsed_ms:.2f}ms"
    )

    return cylinder_mesh


def assemble_model(
    funnel: stl_mesh.Mesh, sleeve: stl_mesh.Mesh, params: FunnelParams = None
) -> stl_mesh.Mesh:
    """
    Combine funnel and cylinder sleeve into a single mesh.

    Concatenates face data arrays from both component meshes into one
    unified numpy-stl Mesh object. Both meshes are generated with matching
    circumferential angles and share the same radius at the junction
    (funnel top outer = sleeve inner), so they align perfectly.

    If params is provided, performs a junction alignment check: extracts
    vertices from the funnel top outer ring and the sleeve inner ring at
    Z = funnel_height, then computes the maximum positional misalignment.
    If misalignment exceeds 1e-4 mm, logs a warning.

    Args:
        funnel: numpy-stl Mesh for the hollow truncated cone.
        sleeve: numpy-stl Mesh for the hollow cylinder sleeve.
        params: Optional FunnelParams used for junction gap detection logging.

    Returns:
        A numpy-stl Mesh combining both components.
    """
    start_time = time.perf_counter()
    logger.info(
        "assemble_model() entry - "
        f"funnel_faces={len(funnel.data)}, sleeve_faces={len(sleeve.data)}"
    )

    # Concatenate face data arrays from both meshes
    combined_data = np.concatenate([funnel.data, sleeve.data])
    combined_mesh = stl_mesh.Mesh(combined_data)

    total_faces = len(combined_data)
    logger.info(
        f"assemble_model() combined mesh created - total_faces={total_faces}"
    )

    # Junction gap detection (optional, requires params)
    if params is not None:
        _check_junction_alignment(combined_mesh, params)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        f"assemble_model() exit - "
        f"total_faces={total_faces}, elapsed={elapsed_ms:.2f}ms"
    )

    return combined_mesh


def _check_junction_alignment(
    combined_mesh: stl_mesh.Mesh, params: FunnelParams
) -> None:
    """
    Check vertex alignment at the funnel-sleeve junction boundary.

    The funnel top outer ring and the sleeve inner ring at Z = funnel_height
    should share the same circle (radius = top_diameter / 2). This function
    extracts vertices near that junction plane and checks whether the maximum
    positional deviation exceeds a tolerance of 1e-4 mm.

    Args:
        combined_mesh: The assembled mesh to inspect.
        params: FunnelParams for computing expected junction geometry.
    """
    tolerance = 1e-4  # mm
    z_junction = params.funnel_height
    expected_radius = params.top_diameter / 2.0

    # Extract all unique vertices from the combined mesh at the junction Z level
    # Each face has 3 vertices stored in vectors (shape: [N, 3, 3])
    all_vectors = combined_mesh.vectors  # shape: (num_faces, 3, 3)
    all_vertices = all_vectors.reshape(-1, 3)  # shape: (num_faces*3, 3)

    # Find vertices at the junction Z plane (within tolerance)
    z_mask = np.abs(all_vertices[:, 2] - z_junction) < tolerance
    junction_vertices = all_vertices[z_mask]

    if len(junction_vertices) == 0:
        logger.debug(
            "_check_junction_alignment() - no vertices found at junction "
            f"Z={z_junction:.4f}mm"
        )
        return

    # Compute radial distance from Z-axis for each junction vertex
    radii = np.sqrt(
        junction_vertices[:, 0] ** 2 + junction_vertices[:, 1] ** 2
    )

    # Filter vertices near the expected junction radius (the shared circle)
    # Use a tight band around the expected radius to exclude inner-wall vertices
    radius_band = max(tolerance * 10, 0.01)  # small band around expected radius
    radius_mask = np.abs(radii - expected_radius) < radius_band
    junction_ring_vertices = junction_vertices[radius_mask]

    if len(junction_ring_vertices) == 0:
        logger.debug(
            "_check_junction_alignment() - no vertices at expected radius "
            f"{expected_radius:.4f}mm at Z={z_junction:.4f}mm"
        )
        return

    # Compute max deviation from expected radius at junction
    junction_radii = np.sqrt(
        junction_ring_vertices[:, 0] ** 2 + junction_ring_vertices[:, 1] ** 2
    )
    max_radial_deviation = np.max(np.abs(junction_radii - expected_radius))

    # Also check Z deviation from junction plane
    max_z_deviation = np.max(
        np.abs(junction_ring_vertices[:, 2] - z_junction)
    )

    max_misalignment = max(max_radial_deviation, max_z_deviation)

    if max_misalignment > tolerance:
        logger.warning(
            f"Junction misalignment detected: max_deviation={max_misalignment:.6f}mm "
            f"(tolerance={tolerance}mm) at Z={z_junction:.4f}mm, "
            f"expected_radius={expected_radius:.4f}mm"
        )
    else:
        logger.debug(
            f"_check_junction_alignment() - alignment OK, "
            f"max_deviation={max_misalignment:.6f}mm (within tolerance={tolerance}mm)"
        )
