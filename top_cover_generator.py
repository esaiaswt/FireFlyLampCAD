"""Top cover mesh generation module.

Generates a circular top cover consisting of:
- A cylindrical wall (hollow cylinder) with configurable inner diameter,
  wall thickness, and height.
- A flat top surface with a diamond/rhombus wire grid pattern for maximum
  airflow while preventing objects from passing through.

The top surface is a grid of thin bars crossing at 45 degrees, creating
diamond-shaped openings. Only the portions of bars within the circular
disc boundary are generated.

Uses numpy-stl for mesh creation with NumPy-backed vectorized operations.
"""

import time
import math
import numpy as np
from stl import mesh as stl_mesh

from logging_config import get_logger

logger = get_logger(__name__)


def generate_top_cover_mesh(
    inner_diameter: float = 92.0,
    wall_thickness: float = 1.2,
    cover_height: float = 10.0,
    grid_spacing: float = 20.0,
    bar_width: float = 1.2,
    num_segments: int = 64,
) -> stl_mesh.Mesh:
    """Generate the complete top cover mesh with diamond grid pattern.

    The cover is a short cylinder (like a lid) with:
    - Outer diameter = inner_diameter + 2 * wall_thickness
    - Height = cover_height
    - Top face is a diamond wire grid (bars crossing at 45 degrees)

    The diamond opening diagonal = grid_spacing * sqrt(2). To block 40mm
    objects, use grid_spacing ~= 20mm (diagonal ~28mm).

    Coordinate system:
    - Center at X=0, Y=0
    - Z=0 is the bottom of the cover
    - Z=cover_height is the top

    Args:
        inner_diameter: Inner diameter of the cylindrical wall in mm.
        wall_thickness: Wall thickness of the cylinder in mm.
        cover_height: Height of the cylindrical cover in mm.
        grid_spacing: Distance between parallel bars in mm (controls opening size).
        bar_width: Width of each grid bar in mm.
        num_segments: Number of circumferential segments for the cylinder wall.

    Returns:
        A numpy-stl Mesh object representing the complete top cover.
    """
    start_time = time.perf_counter()
    logger.info(
        "generate_top_cover_mesh() entry - "
        f"inner_diameter={inner_diameter}, "
        f"wall_thickness={wall_thickness}, "
        f"cover_height={cover_height}, "
        f"grid_spacing={grid_spacing}, "
        f"bar_width={bar_width}, "
        f"num_segments={num_segments}"
    )

    inner_radius = inner_diameter / 2.0
    outer_radius = inner_radius + wall_thickness

    # Generate cylinder wall (hollow cylinder)
    wall_faces = _generate_cylinder_wall(
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        height=cover_height,
        num_segments=num_segments,
    )
    logger.debug(f"Cylinder wall: {len(wall_faces)} faces")

    # Generate bottom annular disc (solid, no holes)
    bottom_faces = _generate_annular_disc(
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        z=0.0,
        num_segments=num_segments,
        flip_normals=True,
    )
    logger.debug(f"Bottom disc: {len(bottom_faces)} faces")

    # Generate top diamond grid surface
    grid_faces = _generate_diamond_grid(
        outer_radius=outer_radius,
        z_top=cover_height,
        bar_width=bar_width,
        grid_spacing=grid_spacing,
        disc_thickness=wall_thickness,
    )
    logger.debug(f"Diamond grid: {len(grid_faces)} faces")

    # Generate a rim ring at the top edge to connect grid to cylinder wall
    rim_faces = _generate_annular_disc(
        inner_radius=outer_radius - wall_thickness,
        outer_radius=outer_radius,
        z=cover_height,
        num_segments=num_segments,
        flip_normals=False,
    )
    logger.debug(f"Top rim: {len(rim_faces)} faces")

    # Combine all faces
    all_faces = np.concatenate([wall_faces, bottom_faces, grid_faces, rim_faces], axis=0)

    # Create numpy-stl mesh
    cover_mesh = stl_mesh.Mesh(np.zeros(len(all_faces), dtype=stl_mesh.Mesh.dtype))
    cover_mesh.vectors = all_faces

    elapsed = (time.perf_counter() - start_time) * 1000
    logger.info(
        f"generate_top_cover_mesh() exit - "
        f"total_faces={len(all_faces)}, elapsed={elapsed:.2f}ms"
    )

    return cover_mesh


def _generate_cylinder_wall(
    inner_radius: float,
    outer_radius: float,
    height: float,
    num_segments: int,
) -> np.ndarray:
    """Generate a hollow cylinder wall (outer + inner surfaces).

    Args:
        inner_radius: Inner radius of the cylinder.
        outer_radius: Outer radius of the cylinder.
        height: Height of the cylinder.
        num_segments: Number of circumferential segments.

    Returns:
        Numpy array of shape (N, 3, 3) containing triangle vertices.
    """
    logger.debug(
        f"_generate_cylinder_wall() - inner_r={inner_radius:.2f}, "
        f"outer_r={outer_radius:.2f}, height={height:.2f}"
    )

    n = num_segments
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    total_faces = 4 * n
    faces = np.zeros((total_faces, 3, 3), dtype=np.float64)
    idx = 0

    z_bottom = 0.0
    z_top = height

    # Outer surface (normals pointing outward)
    for i in range(n):
        i_next = (i + 1) % n
        faces[idx] = [
            [outer_radius * cos_a[i], outer_radius * sin_a[i], z_bottom],
            [outer_radius * cos_a[i_next], outer_radius * sin_a[i_next], z_bottom],
            [outer_radius * cos_a[i_next], outer_radius * sin_a[i_next], z_top],
        ]
        idx += 1
        faces[idx] = [
            [outer_radius * cos_a[i], outer_radius * sin_a[i], z_bottom],
            [outer_radius * cos_a[i_next], outer_radius * sin_a[i_next], z_top],
            [outer_radius * cos_a[i], outer_radius * sin_a[i], z_top],
        ]
        idx += 1

    # Inner surface (normals pointing inward)
    for i in range(n):
        i_next = (i + 1) % n
        faces[idx] = [
            [inner_radius * cos_a[i], inner_radius * sin_a[i], z_bottom],
            [inner_radius * cos_a[i_next], inner_radius * sin_a[i_next], z_top],
            [inner_radius * cos_a[i_next], inner_radius * sin_a[i_next], z_bottom],
        ]
        idx += 1
        faces[idx] = [
            [inner_radius * cos_a[i], inner_radius * sin_a[i], z_bottom],
            [inner_radius * cos_a[i], inner_radius * sin_a[i], z_top],
            [inner_radius * cos_a[i_next], inner_radius * sin_a[i_next], z_top],
        ]
        idx += 1

    return faces


def _generate_annular_disc(
    inner_radius: float,
    outer_radius: float,
    z: float,
    num_segments: int,
    flip_normals: bool = False,
) -> np.ndarray:
    """Generate a flat annular disc (ring) at a given Z height.

    Args:
        inner_radius: Inner radius of the annular disc.
        outer_radius: Outer radius of the annular disc.
        z: Z-coordinate of the disc.
        num_segments: Number of circumferential segments.
        flip_normals: If True, normals face downward (-Z); otherwise upward (+Z).

    Returns:
        Numpy array of shape (N, 3, 3) containing triangle vertices.
    """
    n = num_segments
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    faces = np.zeros((2 * n, 3, 3), dtype=np.float64)
    idx = 0

    for i in range(n):
        i_next = (i + 1) % n
        if flip_normals:
            faces[idx] = [
                [inner_radius * cos_a[i], inner_radius * sin_a[i], z],
                [outer_radius * cos_a[i_next], outer_radius * sin_a[i_next], z],
                [outer_radius * cos_a[i], outer_radius * sin_a[i], z],
            ]
            idx += 1
            faces[idx] = [
                [inner_radius * cos_a[i], inner_radius * sin_a[i], z],
                [inner_radius * cos_a[i_next], inner_radius * sin_a[i_next], z],
                [outer_radius * cos_a[i_next], outer_radius * sin_a[i_next], z],
            ]
            idx += 1
        else:
            faces[idx] = [
                [inner_radius * cos_a[i], inner_radius * sin_a[i], z],
                [outer_radius * cos_a[i], outer_radius * sin_a[i], z],
                [outer_radius * cos_a[i_next], outer_radius * sin_a[i_next], z],
            ]
            idx += 1
            faces[idx] = [
                [inner_radius * cos_a[i], inner_radius * sin_a[i], z],
                [outer_radius * cos_a[i_next], outer_radius * sin_a[i_next], z],
                [inner_radius * cos_a[i_next], inner_radius * sin_a[i_next], z],
            ]
            idx += 1

    return faces


def _generate_diamond_grid(
    outer_radius: float,
    z_top: float,
    bar_width: float,
    grid_spacing: float,
    disc_thickness: float,
) -> np.ndarray:
    """Generate a diamond/rhombus wire grid on the top surface.

    Creates two sets of parallel bars crossing at 45 and -45 degrees,
    forming diamond-shaped openings. Each bar is a thin rectangular
    solid (extruded along its length) clipped to the circular boundary.

    The diamond openings have a diagonal = grid_spacing * sqrt(2).
    For blocking 40mm objects, grid_spacing ~= 20mm gives ~28mm diagonal.

    Args:
        outer_radius: Radius of the circular disc to fill with grid.
        z_top: Z-coordinate of the top of the grid.
        bar_width: Width (thickness) of each bar in mm.
        grid_spacing: Distance between parallel bars in mm.
        disc_thickness: Vertical thickness of the grid bars.

    Returns:
        Numpy array of shape (N, 3, 3) containing triangle vertices.
    """
    z_bottom = z_top - disc_thickness
    half_bar = bar_width / 2.0

    all_bar_faces = []

    # Generate bars in two diagonal directions: +45 deg and -45 deg
    # Direction 1: lines going from bottom-left to top-right (slope = +1)
    # Direction 2: lines going from top-left to bottom-right (slope = -1)

    # For a 45-degree grid, parallel lines are spaced grid_spacing apart
    # along the perpendicular direction.
    # Perpendicular spacing for 45-degree lines = grid_spacing

    # Number of lines needed to cover the disc
    # Lines are at perpendicular distance = n * grid_spacing from center
    max_offset = outer_radius + grid_spacing  # ensure coverage to edges

    # Direction 1: bars at 45 degrees (dx=1, dy=1 normalized)
    offsets_1 = _compute_line_offsets(grid_spacing, max_offset)
    for offset in offsets_1:
        bar = _generate_single_bar_45(
            offset=offset,
            angle_deg=45.0,
            outer_radius=outer_radius,
            half_bar=half_bar,
            z_top=z_top,
            z_bottom=z_bottom,
        )
        if bar is not None and len(bar) > 0:
            all_bar_faces.append(bar)

    # Direction 2: bars at -45 degrees (dx=1, dy=-1 normalized)
    offsets_2 = _compute_line_offsets(grid_spacing, max_offset)
    for offset in offsets_2:
        bar = _generate_single_bar_45(
            offset=offset,
            angle_deg=-45.0,
            outer_radius=outer_radius,
            half_bar=half_bar,
            z_top=z_top,
            z_bottom=z_bottom,
        )
        if bar is not None and len(bar) > 0:
            all_bar_faces.append(bar)

    if not all_bar_faces:
        return np.zeros((0, 3, 3), dtype=np.float64)

    return np.concatenate(all_bar_faces, axis=0)


def _compute_line_offsets(grid_spacing: float, max_offset: float) -> list:
    """Compute the perpendicular offsets for grid lines.

    Returns offsets from -max_offset to +max_offset at grid_spacing intervals,
    including the center line (offset=0).

    Args:
        grid_spacing: Distance between parallel lines.
        max_offset: Maximum offset from center.

    Returns:
        List of float offsets.
    """
    offsets = []
    offset = 0.0
    while offset <= max_offset:
        offsets.append(offset)
        if offset != 0.0:
            offsets.append(-offset)
        offset += grid_spacing
    return offsets


def _generate_single_bar_45(
    offset: float,
    angle_deg: float,
    outer_radius: float,
    half_bar: float,
    z_top: float,
    z_bottom: float,
) -> np.ndarray:
    """Generate a single bar of the grid, clipped to the circular boundary.

    The bar is an infinite line at the given angle, offset perpendicularly
    from the center. It is clipped to the circle (only the chord inside
    the disc is generated).

    Each bar is a 3D rectangular solid: length along the chord, width = 2*half_bar,
    height from z_bottom to z_top. It has 6 faces (top, bottom, 4 sides) = 12 triangles.

    Args:
        offset: Perpendicular distance from center to the line.
        angle_deg: Angle of the bar in degrees (45 or -45).
        outer_radius: Radius of the circular disc.
        half_bar: Half the bar width.
        z_top: Top Z of the bar.
        z_bottom: Bottom Z of the bar.

    Returns:
        Numpy array of shape (N, 3, 3) containing triangle vertices, or None.
    """
    # Check if line intersects the circle at all
    if abs(offset) >= outer_radius:
        return None

    # Compute chord length: for a line at perpendicular distance d from center,
    # chord half-length = sqrt(r^2 - d^2)
    chord_half = math.sqrt(outer_radius * outer_radius - offset * offset)

    if chord_half < half_bar:
        return None  # Too short to matter

    # Bar direction unit vector
    angle_rad = math.radians(angle_deg)
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)

    # Perpendicular direction (offset direction)
    # For a line at angle theta, perpendicular is at theta + 90
    px = -dy  # perpendicular x
    py = dx   # perpendicular y

    # Center of the chord
    cx = offset * px
    cy = offset * py

    # Bar endpoints along the chord
    # Start and end points (clipped to circle)
    x_start = cx - chord_half * dx
    y_start = cy - chord_half * dy
    x_end = cx + chord_half * dx
    y_end = cy + chord_half * dy

    # The bar is a rectangle with width = 2 * half_bar, extruded in Z
    # 4 corner points at top and bottom:
    # Offset perpendicular to the bar direction by +/- half_bar
    # Top face corners
    t0 = [x_start + half_bar * px, y_start + half_bar * py, z_top]
    t1 = [x_start - half_bar * px, y_start - half_bar * py, z_top]
    t2 = [x_end - half_bar * px, y_end - half_bar * py, z_top]
    t3 = [x_end + half_bar * px, y_end + half_bar * py, z_top]

    # Bottom face corners
    b0 = [x_start + half_bar * px, y_start + half_bar * py, z_bottom]
    b1 = [x_start - half_bar * px, y_start - half_bar * py, z_bottom]
    b2 = [x_end - half_bar * px, y_end - half_bar * py, z_bottom]
    b3 = [x_end + half_bar * px, y_end + half_bar * py, z_bottom]

    # Generate 12 triangles (6 quads = 12 tris)
    faces = np.zeros((12, 3, 3), dtype=np.float64)

    # Top face (t0, t1, t2, t3) — normal up
    faces[0] = [t0, t3, t2]
    faces[1] = [t0, t2, t1]

    # Bottom face (b0, b1, b2, b3) — normal down
    faces[2] = [b0, b1, b2]
    faces[3] = [b0, b2, b3]

    # Front side (t0, t3, b3, b0) — normal in +perp direction
    faces[4] = [t0, b0, b3]
    faces[5] = [t0, b3, t3]

    # Back side (t1, t2, b2, b1) — normal in -perp direction
    faces[6] = [t1, t2, b2]
    faces[7] = [t1, b2, b1]

    # Left end (t0, t1, b1, b0) — normal toward -dir
    faces[8] = [t0, t1, b1]
    faces[9] = [t0, b1, b0]

    # Right end (t3, t2, b2, b3) — normal toward +dir
    faces[10] = [t3, b3, b2]
    faces[11] = [t3, b2, t2]

    return faces
