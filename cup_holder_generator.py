"""Cup holder mesh generation module.

Generates a cup holder assembly consisting of:
- An inner ring (cup holder) at the top
- Two inner stands (at angle=pi and angle=0) connecting ring to base plate
- A full O-shaped base plate (complete circle)
- 10mm outward extrusions from each inner stand's outer wall
- Two outer stands (120mm tall, 1.2mm wall) at end of each extrusion
- An outer ring (80mm diameter, 1.2mm wall, 10mm height) at top of outer stands

Uses numpy-stl for mesh creation with NumPy-backed vectorized operations.
"""

import time
import math
import numpy as np
from stl import mesh as stl_mesh

from logging_config import get_logger

logger = get_logger(__name__)


def generate_cup_holder_mesh(
    ring_inner_diameter: float = 62.0,
    ring_wall_thickness: float = 1.2,
    ring_height: float = 15.0,
    stand_wall_thickness: float = 3.0,
    total_height: float = 40.0,
    leg_width: float = 10.0,
    extrusion_length: float = 10.0,
    outer_stand_height: float = 120.0,
    outer_stand_wall: float = 1.2,
    outer_ring_diameter: float = 80.0,
    outer_ring_wall: float = 1.2,
    outer_ring_height: float = 10.0,
    num_segments: int = 64,
) -> stl_mesh.Mesh:
    """Generate the complete cup holder assembly mesh.

    Layout (side view, centered at X=0):
        Outer Ring (80mm dia, at top of outer stands)
        _____|_____          _____|_____
       |  Outer   |        |  Outer   |
       |  Stand   |        |  Stand   |
       |  120mm   |        |  120mm   |
       |__________|        |__________|
            |  extrusion        |  extrusion
            |--10mm--|    |--10mm--|
       Inner Stand        Inner Stand
       (angle=pi)         (angle=0)
            |                   |
       [ Inner Ring 62mm ID, 15mm tall ]
            |                   |
       Inner Stand        Inner Stand
            |                   |
       [====== Base O-Plate ======]

    Coordinate system:
    - Center at X=0, Y=0
    - Z=0 is bottom of base plate
    - Inner stands at angle=pi (-X) and angle=0 (+X)

    Args:
        ring_inner_diameter: Inner diameter of the cup holder ring in mm.
        ring_wall_thickness: Wall thickness of the inner ring in mm.
        ring_height: Height of the inner ring in mm.
        stand_wall_thickness: Wall thickness of the inner stands in mm.
        total_height: Height from ring top to base plate bottom in mm.
        leg_width: Radial width of the base O-plate in mm.
        extrusion_length: Outward extrusion length from stand outer wall in mm.
        outer_stand_height: Height of the outer stands in mm.
        outer_stand_wall: Wall thickness of the outer stands in mm.
        outer_ring_diameter: Diameter of the outer ring in mm.
        outer_ring_wall: Wall thickness of the outer ring in mm.
        outer_ring_height: Height (width) of the outer ring in mm.
        num_segments: Number of circumferential segments for circles.

    Returns:
        A numpy-stl Mesh object representing the complete cup holder assembly.
    """
    start_time = time.perf_counter()
    logger.info(
        "generate_cup_holder_mesh() entry - "
        f"ring_inner_diameter={ring_inner_diameter}, "
        f"ring_wall_thickness={ring_wall_thickness}, "
        f"ring_height={ring_height}, "
        f"stand_wall_thickness={stand_wall_thickness}, "
        f"total_height={total_height}, "
        f"leg_width={leg_width}, "
        f"extrusion_length={extrusion_length}, "
        f"outer_stand_height={outer_stand_height}, "
        f"outer_stand_wall={outer_stand_wall}, "
        f"outer_ring_diameter={outer_ring_diameter}, "
        f"outer_ring_wall={outer_ring_wall}, "
        f"outer_ring_height={outer_ring_height}, "
        f"num_segments={num_segments}"
    )

    # Compute derived dimensions
    ring_inner_radius = ring_inner_diameter / 2.0
    ring_outer_radius = ring_inner_radius + ring_wall_thickness
    stand_outer_radius = ring_inner_radius + stand_wall_thickness

    # Base plate: full O, outer aligned with stand outer wall
    plate_outer_radius = stand_outer_radius
    plate_inner_radius = plate_outer_radius - leg_width
    plate_thickness = ring_wall_thickness

    # Extrusion: horizontal plate from stand outer wall outward
    extrusion_outer_radius = stand_outer_radius + extrusion_length

    # Overall height from base to top = 120mm (total_height param)
    z_plate_bottom = 0.0
    z_plate_top = plate_thickness

    # Outer stands: full height from base to top (120mm)
    z_outer_stand_bottom = z_plate_bottom
    z_outer_stand_top = total_height

    # Outer ring at top (top 10mm = outer_ring_height)
    z_outer_ring_top = total_height
    z_outer_ring_bottom = total_height - outer_ring_height

    # Inner ring: positioned in the lower portion of the assembly
    # The inner ring top is at the inner assembly height (total_height parameter
    # in the old sense was 40mm). We'll place the inner ring centered vertically
    # at a reasonable position. Let's keep it at the same relative position:
    # inner ring top = total_height * (40/130) roughly, but let's just use
    # a fixed inner_ring_top_z based on the inner assembly being in the bottom section.
    # Actually, looking at the image, the inner ring is in the lower third.
    # Let's place inner ring top at plate_top + (stand connects plate to ring)
    # We'll use: inner ring top = base plate top + stand_height_below_ring + ring_height
    # For now, keep inner assembly height as a fraction. The inner ring should be
    # below the outer ring. Let's place it at Z=40 (same as before) since that
    # looked correct in the image.
    inner_assembly_height = 40.0  # inner ring top at Z=40
    z_ring_top = inner_assembly_height
    z_ring_bottom = z_ring_top - ring_height

    # Outer ring radii: should encompass the outer stands
    outer_ring_inner_radius = outer_ring_diameter / 2.0
    outer_ring_outer_radius = outer_ring_inner_radius + outer_ring_wall

    logger.debug(
        f"Derived: ring_inner_r={ring_inner_radius:.2f}, "
        f"stand_outer_r={stand_outer_radius:.2f}, "
        f"plate_inner_r={plate_inner_radius:.2f}, "
        f"extrusion_outer_r={extrusion_outer_radius:.2f}, "
        f"z_ring_top={z_ring_top:.2f}, z_ring_bottom={z_ring_bottom:.2f}, "
        f"z_outer_stand_top={z_outer_stand_top:.2f}, "
        f"z_outer_ring_top={z_outer_ring_top:.2f}"
    )

    all_face_arrays = []

    # 1. Inner ring (cup holder)
    all_face_arrays.append(_generate_ring(
        ring_inner_radius, ring_outer_radius,
        z_ring_bottom, z_ring_top, num_segments
    ))

    # 2. Two inner stands (at angle=pi and angle=0) - from plate top to inner ring top
    for center_angle in [math.pi, 0.0]:
        all_face_arrays.append(_generate_stand(
            ring_inner_radius, stand_wall_thickness,
            z_plate_top, z_ring_top, center_angle, num_segments
        ))

    # 3. Full O base plate (complete circle)
    all_face_arrays.append(_generate_ring(
        plate_inner_radius, plate_outer_radius,
        z_plate_bottom, z_plate_top, num_segments
    ))

    # 4. Outward extrusions from each stand (horizontal plates at base level)
    for center_angle in [math.pi, 0.0]:
        all_face_arrays.append(_generate_extrusion(
            stand_outer_radius, extrusion_outer_radius,
            stand_wall_thickness, ring_inner_radius,
            z_plate_bottom, z_plate_top,
            center_angle, num_segments
        ))

    # 5. Two outer stands - from base to outer ring bottom
    for center_angle in [math.pi, 0.0]:
        all_face_arrays.append(_generate_outer_stand(
            extrusion_outer_radius, outer_stand_wall,
            stand_wall_thickness, ring_inner_radius,
            z_outer_stand_bottom, z_outer_stand_top,
            center_angle, num_segments
        ))

    # 6. Outer ring at top of outer stands (connected - ring bottom = stand top)
    all_face_arrays.append(_generate_ring(
        outer_ring_inner_radius, outer_ring_outer_radius,
        z_outer_ring_bottom, z_outer_ring_top, num_segments
    ))

    # 7. Connecting walls between inner stand outer wall and outer stand inner wall
    # These are radial flat walls running from stand_outer_radius to extrusion_outer_radius
    # at the same arc span as the stands, from plate top to inner ring top height
    for center_angle in [math.pi, 0.0]:
        all_face_arrays.append(_generate_connecting_wall(
            stand_outer_radius, extrusion_outer_radius,
            stand_wall_thickness, ring_inner_radius,
            z_plate_top, z_ring_top,
            center_angle, num_segments
        ))

    # 8. Inward extrusions at top of each outer stand connecting to outer ring
    # These bridge from the outer stand inner wall (extrusion_outer_radius) inward
    # to the outer ring outer wall (outer_ring_outer_radius), at the top 10mm
    for center_angle in [math.pi, 0.0]:
        all_face_arrays.append(_generate_extrusion(
            outer_ring_outer_radius, extrusion_outer_radius,
            stand_wall_thickness, ring_inner_radius,
            z_outer_ring_bottom, z_outer_ring_top,
            center_angle, num_segments
        ))

    # Combine all face arrays
    total_faces = sum(len(f) for f in all_face_arrays)
    combined_data = np.zeros(total_faces, dtype=stl_mesh.Mesh.dtype)

    offset = 0
    for face_array in all_face_arrays:
        n = len(face_array)
        if n > 0:
            combined_data['vectors'][offset:offset + n] = face_array
            offset += n

    combined_mesh = stl_mesh.Mesh(combined_data)
    combined_mesh.update_normals()

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        f"generate_cup_holder_mesh() exit - "
        f"total_faces={total_faces}, elapsed={elapsed_ms:.2f}ms"
    )

    return combined_mesh


def _generate_ring(
    inner_radius: float,
    outer_radius: float,
    z_bottom: float,
    z_top: float,
    num_segments: int,
) -> np.ndarray:
    """Generate a full cylindrical ring (hollow cylinder).

    Creates outer surface, inner surface, top annular ring, and bottom annular ring.

    Returns:
        Numpy array of shape (N, 3, 3) containing triangle vertices.
    """
    logger.debug(
        f"_generate_ring() - inner_r={inner_radius:.2f}, outer_r={outer_radius:.2f}, "
        f"z_bottom={z_bottom:.2f}, z_top={z_top:.2f}"
    )

    n = num_segments
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    total_faces = 8 * n
    faces = np.zeros((total_faces, 3, 3), dtype=np.float64)
    idx = 0

    # Outer surface
    for i in range(n):
        i_next = (i + 1) % n
        faces[idx] = [[outer_radius * cos_a[i], outer_radius * sin_a[i], z_bottom],
                      [outer_radius * cos_a[i_next], outer_radius * sin_a[i_next], z_bottom],
                      [outer_radius * cos_a[i_next], outer_radius * sin_a[i_next], z_top]]
        idx += 1
        faces[idx] = [[outer_radius * cos_a[i], outer_radius * sin_a[i], z_bottom],
                      [outer_radius * cos_a[i_next], outer_radius * sin_a[i_next], z_top],
                      [outer_radius * cos_a[i], outer_radius * sin_a[i], z_top]]
        idx += 1

    # Inner surface
    for i in range(n):
        i_next = (i + 1) % n
        faces[idx] = [[inner_radius * cos_a[i], inner_radius * sin_a[i], z_bottom],
                      [inner_radius * cos_a[i_next], inner_radius * sin_a[i_next], z_top],
                      [inner_radius * cos_a[i_next], inner_radius * sin_a[i_next], z_bottom]]
        idx += 1
        faces[idx] = [[inner_radius * cos_a[i], inner_radius * sin_a[i], z_bottom],
                      [inner_radius * cos_a[i], inner_radius * sin_a[i], z_top],
                      [inner_radius * cos_a[i_next], inner_radius * sin_a[i_next], z_top]]
        idx += 1

    # Top annular ring
    for i in range(n):
        i_next = (i + 1) % n
        faces[idx] = [[inner_radius * cos_a[i], inner_radius * sin_a[i], z_top],
                      [outer_radius * cos_a[i], outer_radius * sin_a[i], z_top],
                      [outer_radius * cos_a[i_next], outer_radius * sin_a[i_next], z_top]]
        idx += 1
        faces[idx] = [[inner_radius * cos_a[i], inner_radius * sin_a[i], z_top],
                      [outer_radius * cos_a[i_next], outer_radius * sin_a[i_next], z_top],
                      [inner_radius * cos_a[i_next], inner_radius * sin_a[i_next], z_top]]
        idx += 1

    # Bottom annular ring
    for i in range(n):
        i_next = (i + 1) % n
        faces[idx] = [[inner_radius * cos_a[i], inner_radius * sin_a[i], z_bottom],
                      [outer_radius * cos_a[i_next], outer_radius * sin_a[i_next], z_bottom],
                      [outer_radius * cos_a[i], outer_radius * sin_a[i], z_bottom]]
        idx += 1
        faces[idx] = [[inner_radius * cos_a[i], inner_radius * sin_a[i], z_bottom],
                      [inner_radius * cos_a[i_next], inner_radius * sin_a[i_next], z_bottom],
                      [outer_radius * cos_a[i_next], outer_radius * sin_a[i_next], z_bottom]]
        idx += 1

    return faces


def _generate_stand(
    ring_inner_radius: float,
    stand_wall_thickness: float,
    z_bottom: float,
    z_top: float,
    center_angle: float,
    num_segments: int,
) -> np.ndarray:
    """Generate a vertical stand (arc segment) at a given angle.

    The stand is an arc-shaped vertical wall centered at center_angle.

    Args:
        ring_inner_radius: Inner radius of the stand arc.
        stand_wall_thickness: Radial thickness of the stand.
        z_bottom: Bottom Z coordinate.
        z_top: Top Z coordinate.
        center_angle: Angular center of the stand (radians).
        num_segments: Total segments for full circle (used to proportion).

    Returns:
        Numpy array of shape (N, 3, 3) containing triangle vertices.
    """
    stand_arc_half_angle = math.atan2(stand_wall_thickness * 2, ring_inner_radius)
    stand_arc_half_angle = max(stand_arc_half_angle, math.radians(15))

    inner_r = ring_inner_radius
    outer_r = ring_inner_radius + stand_wall_thickness

    stand_segments = max(8, int(num_segments * (2 * stand_arc_half_angle) / (2 * np.pi)))

    start_angle = center_angle - stand_arc_half_angle
    end_angle = center_angle + stand_arc_half_angle

    angles = np.linspace(start_angle, end_angle, stand_segments + 1)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    n = stand_segments
    total_faces = 8 * n + 4
    faces = np.zeros((total_faces, 3, 3), dtype=np.float64)
    idx = 0

    # Outer surface
    for i in range(n):
        faces[idx] = [[outer_r * cos_a[i], outer_r * sin_a[i], z_bottom],
                      [outer_r * cos_a[i + 1], outer_r * sin_a[i + 1], z_bottom],
                      [outer_r * cos_a[i + 1], outer_r * sin_a[i + 1], z_top]]
        idx += 1
        faces[idx] = [[outer_r * cos_a[i], outer_r * sin_a[i], z_bottom],
                      [outer_r * cos_a[i + 1], outer_r * sin_a[i + 1], z_top],
                      [outer_r * cos_a[i], outer_r * sin_a[i], z_top]]
        idx += 1

    # Inner surface
    for i in range(n):
        faces[idx] = [[inner_r * cos_a[i], inner_r * sin_a[i], z_bottom],
                      [inner_r * cos_a[i + 1], inner_r * sin_a[i + 1], z_top],
                      [inner_r * cos_a[i + 1], inner_r * sin_a[i + 1], z_bottom]]
        idx += 1
        faces[idx] = [[inner_r * cos_a[i], inner_r * sin_a[i], z_bottom],
                      [inner_r * cos_a[i], inner_r * sin_a[i], z_top],
                      [inner_r * cos_a[i + 1], inner_r * sin_a[i + 1], z_top]]
        idx += 1

    # Top annular
    for i in range(n):
        faces[idx] = [[inner_r * cos_a[i], inner_r * sin_a[i], z_top],
                      [outer_r * cos_a[i], outer_r * sin_a[i], z_top],
                      [outer_r * cos_a[i + 1], outer_r * sin_a[i + 1], z_top]]
        idx += 1
        faces[idx] = [[inner_r * cos_a[i], inner_r * sin_a[i], z_top],
                      [outer_r * cos_a[i + 1], outer_r * sin_a[i + 1], z_top],
                      [inner_r * cos_a[i + 1], inner_r * sin_a[i + 1], z_top]]
        idx += 1

    # Bottom annular
    for i in range(n):
        faces[idx] = [[inner_r * cos_a[i], inner_r * sin_a[i], z_bottom],
                      [outer_r * cos_a[i + 1], outer_r * sin_a[i + 1], z_bottom],
                      [outer_r * cos_a[i], outer_r * sin_a[i], z_bottom]]
        idx += 1
        faces[idx] = [[inner_r * cos_a[i], inner_r * sin_a[i], z_bottom],
                      [inner_r * cos_a[i + 1], inner_r * sin_a[i + 1], z_bottom],
                      [outer_r * cos_a[i + 1], outer_r * sin_a[i + 1], z_bottom]]
        idx += 1

    # Left end cap
    faces[idx] = [[inner_r * cos_a[0], inner_r * sin_a[0], z_bottom],
                  [outer_r * cos_a[0], outer_r * sin_a[0], z_bottom],
                  [outer_r * cos_a[0], outer_r * sin_a[0], z_top]]
    idx += 1
    faces[idx] = [[inner_r * cos_a[0], inner_r * sin_a[0], z_bottom],
                  [outer_r * cos_a[0], outer_r * sin_a[0], z_top],
                  [inner_r * cos_a[0], inner_r * sin_a[0], z_top]]
    idx += 1

    # Right end cap
    faces[idx] = [[inner_r * cos_a[-1], inner_r * sin_a[-1], z_bottom],
                  [outer_r * cos_a[-1], outer_r * sin_a[-1], z_top],
                  [outer_r * cos_a[-1], outer_r * sin_a[-1], z_bottom]]
    idx += 1
    faces[idx] = [[inner_r * cos_a[-1], inner_r * sin_a[-1], z_bottom],
                  [inner_r * cos_a[-1], inner_r * sin_a[-1], z_top],
                  [outer_r * cos_a[-1], outer_r * sin_a[-1], z_top]]
    idx += 1

    return faces[:idx]


def _generate_extrusion(
    inner_radius: float,
    outer_radius: float,
    stand_wall_thickness: float,
    ring_inner_radius: float,
    z_bottom: float,
    z_top: float,
    center_angle: float,
    num_segments: int,
) -> np.ndarray:
    """Generate a horizontal extrusion plate from the stand outer wall outward.

    This is a flat plate (same arc span as the stand) extending radially
    outward from stand_outer_radius to extrusion_outer_radius.

    Args:
        inner_radius: Inner radius of extrusion (= stand outer radius).
        outer_radius: Outer radius of extrusion.
        stand_wall_thickness: Stand wall thickness (for arc calculation).
        ring_inner_radius: Inner ring radius (for arc calculation).
        z_bottom: Bottom Z.
        z_top: Top Z.
        center_angle: Angular center of the extrusion.
        num_segments: Segments for full circle.

    Returns:
        Numpy array of shape (N, 3, 3) containing triangle vertices.
    """
    stand_arc_half_angle = math.atan2(stand_wall_thickness * 2, ring_inner_radius)
    stand_arc_half_angle = max(stand_arc_half_angle, math.radians(15))

    start_angle = center_angle - stand_arc_half_angle
    end_angle = center_angle + stand_arc_half_angle

    arc_segments = max(4, int(num_segments * (2 * stand_arc_half_angle) / (2 * np.pi)))
    angles = np.linspace(start_angle, end_angle, arc_segments + 1)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    n = arc_segments
    # 4 surfaces + 2 end caps
    total_faces = 8 * n + 4
    faces = np.zeros((total_faces, 3, 3), dtype=np.float64)
    idx = 0

    # Outer curved surface
    for i in range(n):
        faces[idx] = [[outer_radius * cos_a[i], outer_radius * sin_a[i], z_bottom],
                      [outer_radius * cos_a[i + 1], outer_radius * sin_a[i + 1], z_bottom],
                      [outer_radius * cos_a[i + 1], outer_radius * sin_a[i + 1], z_top]]
        idx += 1
        faces[idx] = [[outer_radius * cos_a[i], outer_radius * sin_a[i], z_bottom],
                      [outer_radius * cos_a[i + 1], outer_radius * sin_a[i + 1], z_top],
                      [outer_radius * cos_a[i], outer_radius * sin_a[i], z_top]]
        idx += 1

    # Inner curved surface
    for i in range(n):
        faces[idx] = [[inner_radius * cos_a[i], inner_radius * sin_a[i], z_bottom],
                      [inner_radius * cos_a[i + 1], inner_radius * sin_a[i + 1], z_top],
                      [inner_radius * cos_a[i + 1], inner_radius * sin_a[i + 1], z_bottom]]
        idx += 1
        faces[idx] = [[inner_radius * cos_a[i], inner_radius * sin_a[i], z_bottom],
                      [inner_radius * cos_a[i], inner_radius * sin_a[i], z_top],
                      [inner_radius * cos_a[i + 1], inner_radius * sin_a[i + 1], z_top]]
        idx += 1

    # Top surface
    for i in range(n):
        faces[idx] = [[inner_radius * cos_a[i], inner_radius * sin_a[i], z_top],
                      [outer_radius * cos_a[i], outer_radius * sin_a[i], z_top],
                      [outer_radius * cos_a[i + 1], outer_radius * sin_a[i + 1], z_top]]
        idx += 1
        faces[idx] = [[inner_radius * cos_a[i], inner_radius * sin_a[i], z_top],
                      [outer_radius * cos_a[i + 1], outer_radius * sin_a[i + 1], z_top],
                      [inner_radius * cos_a[i + 1], inner_radius * sin_a[i + 1], z_top]]
        idx += 1

    # Bottom surface
    for i in range(n):
        faces[idx] = [[inner_radius * cos_a[i], inner_radius * sin_a[i], z_bottom],
                      [outer_radius * cos_a[i + 1], outer_radius * sin_a[i + 1], z_bottom],
                      [outer_radius * cos_a[i], outer_radius * sin_a[i], z_bottom]]
        idx += 1
        faces[idx] = [[inner_radius * cos_a[i], inner_radius * sin_a[i], z_bottom],
                      [inner_radius * cos_a[i + 1], inner_radius * sin_a[i + 1], z_bottom],
                      [outer_radius * cos_a[i + 1], outer_radius * sin_a[i + 1], z_bottom]]
        idx += 1

    # Left end cap
    faces[idx] = [[inner_radius * cos_a[0], inner_radius * sin_a[0], z_bottom],
                  [outer_radius * cos_a[0], outer_radius * sin_a[0], z_bottom],
                  [outer_radius * cos_a[0], outer_radius * sin_a[0], z_top]]
    idx += 1
    faces[idx] = [[inner_radius * cos_a[0], inner_radius * sin_a[0], z_bottom],
                  [outer_radius * cos_a[0], outer_radius * sin_a[0], z_top],
                  [inner_radius * cos_a[0], inner_radius * sin_a[0], z_top]]
    idx += 1

    # Right end cap
    faces[idx] = [[inner_radius * cos_a[-1], inner_radius * sin_a[-1], z_bottom],
                  [outer_radius * cos_a[-1], outer_radius * sin_a[-1], z_top],
                  [outer_radius * cos_a[-1], outer_radius * sin_a[-1], z_bottom]]
    idx += 1
    faces[idx] = [[inner_radius * cos_a[-1], inner_radius * sin_a[-1], z_bottom],
                  [inner_radius * cos_a[-1], inner_radius * sin_a[-1], z_top],
                  [outer_radius * cos_a[-1], outer_radius * sin_a[-1], z_top]]
    idx += 1

    return faces[:idx]


def _generate_outer_stand(
    extrusion_outer_radius: float,
    outer_stand_wall: float,
    stand_wall_thickness: float,
    ring_inner_radius: float,
    z_bottom: float,
    z_top: float,
    center_angle: float,
    num_segments: int,
) -> np.ndarray:
    """Generate an outer stand at the end of the extrusion.

    The outer stand is a vertical arc-shaped wall at the extrusion outer radius.
    It has the same angular span as the inner stand.

    Args:
        extrusion_outer_radius: Radius where the outer stand starts.
        outer_stand_wall: Wall thickness of the outer stand.
        stand_wall_thickness: Inner stand wall thickness (for arc calculation).
        ring_inner_radius: Inner ring radius (for arc calculation).
        z_bottom: Bottom Z coordinate.
        z_top: Top Z coordinate.
        center_angle: Angular center.
        num_segments: Segments for full circle.

    Returns:
        Numpy array of shape (N, 3, 3) containing triangle vertices.
    """
    stand_arc_half_angle = math.atan2(stand_wall_thickness * 2, ring_inner_radius)
    stand_arc_half_angle = max(stand_arc_half_angle, math.radians(15))

    inner_r = extrusion_outer_radius
    outer_r = extrusion_outer_radius + outer_stand_wall

    arc_segments = max(4, int(num_segments * (2 * stand_arc_half_angle) / (2 * np.pi)))

    start_angle = center_angle - stand_arc_half_angle
    end_angle = center_angle + stand_arc_half_angle

    angles = np.linspace(start_angle, end_angle, arc_segments + 1)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    n = arc_segments
    total_faces = 8 * n + 4
    faces = np.zeros((total_faces, 3, 3), dtype=np.float64)
    idx = 0

    # Outer surface
    for i in range(n):
        faces[idx] = [[outer_r * cos_a[i], outer_r * sin_a[i], z_bottom],
                      [outer_r * cos_a[i + 1], outer_r * sin_a[i + 1], z_bottom],
                      [outer_r * cos_a[i + 1], outer_r * sin_a[i + 1], z_top]]
        idx += 1
        faces[idx] = [[outer_r * cos_a[i], outer_r * sin_a[i], z_bottom],
                      [outer_r * cos_a[i + 1], outer_r * sin_a[i + 1], z_top],
                      [outer_r * cos_a[i], outer_r * sin_a[i], z_top]]
        idx += 1

    # Inner surface
    for i in range(n):
        faces[idx] = [[inner_r * cos_a[i], inner_r * sin_a[i], z_bottom],
                      [inner_r * cos_a[i + 1], inner_r * sin_a[i + 1], z_top],
                      [inner_r * cos_a[i + 1], inner_r * sin_a[i + 1], z_bottom]]
        idx += 1
        faces[idx] = [[inner_r * cos_a[i], inner_r * sin_a[i], z_bottom],
                      [inner_r * cos_a[i], inner_r * sin_a[i], z_top],
                      [inner_r * cos_a[i + 1], inner_r * sin_a[i + 1], z_top]]
        idx += 1

    # Top
    for i in range(n):
        faces[idx] = [[inner_r * cos_a[i], inner_r * sin_a[i], z_top],
                      [outer_r * cos_a[i], outer_r * sin_a[i], z_top],
                      [outer_r * cos_a[i + 1], outer_r * sin_a[i + 1], z_top]]
        idx += 1
        faces[idx] = [[inner_r * cos_a[i], inner_r * sin_a[i], z_top],
                      [outer_r * cos_a[i + 1], outer_r * sin_a[i + 1], z_top],
                      [inner_r * cos_a[i + 1], inner_r * sin_a[i + 1], z_top]]
        idx += 1

    # Bottom
    for i in range(n):
        faces[idx] = [[inner_r * cos_a[i], inner_r * sin_a[i], z_bottom],
                      [outer_r * cos_a[i + 1], outer_r * sin_a[i + 1], z_bottom],
                      [outer_r * cos_a[i], outer_r * sin_a[i], z_bottom]]
        idx += 1
        faces[idx] = [[inner_r * cos_a[i], inner_r * sin_a[i], z_bottom],
                      [inner_r * cos_a[i + 1], inner_r * sin_a[i + 1], z_bottom],
                      [outer_r * cos_a[i + 1], outer_r * sin_a[i + 1], z_bottom]]
        idx += 1

    # Left end cap
    faces[idx] = [[inner_r * cos_a[0], inner_r * sin_a[0], z_bottom],
                  [outer_r * cos_a[0], outer_r * sin_a[0], z_bottom],
                  [outer_r * cos_a[0], outer_r * sin_a[0], z_top]]
    idx += 1
    faces[idx] = [[inner_r * cos_a[0], inner_r * sin_a[0], z_bottom],
                  [outer_r * cos_a[0], outer_r * sin_a[0], z_top],
                  [inner_r * cos_a[0], inner_r * sin_a[0], z_top]]
    idx += 1

    # Right end cap
    faces[idx] = [[inner_r * cos_a[-1], inner_r * sin_a[-1], z_bottom],
                  [outer_r * cos_a[-1], outer_r * sin_a[-1], z_top],
                  [outer_r * cos_a[-1], outer_r * sin_a[-1], z_bottom]]
    idx += 1
    faces[idx] = [[inner_r * cos_a[-1], inner_r * sin_a[-1], z_bottom],
                  [inner_r * cos_a[-1], inner_r * sin_a[-1], z_top],
                  [outer_r * cos_a[-1], outer_r * sin_a[-1], z_top]]
    idx += 1

    return faces[:idx]


def _generate_connecting_wall(
    inner_radius: float,
    outer_radius: float,
    stand_wall_thickness: float,
    ring_inner_radius: float,
    z_bottom: float,
    z_top: float,
    center_angle: float,
    num_segments: int,
) -> np.ndarray:
    """Generate a radial connecting wall between inner stand and outer stand.

    This is a flat vertical wall running radially from the inner stand outer
    wall (stand_outer_radius) to the outer stand inner wall (extrusion_outer_radius),
    with the same angular span as the stands. It bridges the gap between the
    inner and outer stands.

    Args:
        inner_radius: Inner radius of wall (= stand outer radius).
        outer_radius: Outer radius of wall (= extrusion outer radius).
        stand_wall_thickness: Stand wall thickness (for arc calculation).
        ring_inner_radius: Inner ring radius (for arc calculation).
        z_bottom: Bottom Z coordinate.
        z_top: Top Z coordinate.
        center_angle: Angular center.
        num_segments: Segments for full circle.

    Returns:
        Numpy array of shape (N, 3, 3) containing triangle vertices.
    """
    stand_arc_half_angle = math.atan2(stand_wall_thickness * 2, ring_inner_radius)
    stand_arc_half_angle = max(stand_arc_half_angle, math.radians(15))

    start_angle = center_angle - stand_arc_half_angle
    end_angle = center_angle + stand_arc_half_angle

    arc_segments = max(4, int(num_segments * (2 * stand_arc_half_angle) / (2 * np.pi)))
    angles = np.linspace(start_angle, end_angle, arc_segments + 1)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    n = arc_segments
    # This wall has: outer surface, inner surface, top, bottom, 2 end caps
    total_faces = 8 * n + 4
    faces = np.zeros((total_faces, 3, 3), dtype=np.float64)
    idx = 0

    # Outer curved surface (at outer_radius)
    for i in range(n):
        faces[idx] = [[outer_radius * cos_a[i], outer_radius * sin_a[i], z_bottom],
                      [outer_radius * cos_a[i + 1], outer_radius * sin_a[i + 1], z_bottom],
                      [outer_radius * cos_a[i + 1], outer_radius * sin_a[i + 1], z_top]]
        idx += 1
        faces[idx] = [[outer_radius * cos_a[i], outer_radius * sin_a[i], z_bottom],
                      [outer_radius * cos_a[i + 1], outer_radius * sin_a[i + 1], z_top],
                      [outer_radius * cos_a[i], outer_radius * sin_a[i], z_top]]
        idx += 1

    # Inner curved surface (at inner_radius)
    for i in range(n):
        faces[idx] = [[inner_radius * cos_a[i], inner_radius * sin_a[i], z_bottom],
                      [inner_radius * cos_a[i + 1], inner_radius * sin_a[i + 1], z_top],
                      [inner_radius * cos_a[i + 1], inner_radius * sin_a[i + 1], z_bottom]]
        idx += 1
        faces[idx] = [[inner_radius * cos_a[i], inner_radius * sin_a[i], z_bottom],
                      [inner_radius * cos_a[i], inner_radius * sin_a[i], z_top],
                      [inner_radius * cos_a[i + 1], inner_radius * sin_a[i + 1], z_top]]
        idx += 1

    # Top surface
    for i in range(n):
        faces[idx] = [[inner_radius * cos_a[i], inner_radius * sin_a[i], z_top],
                      [outer_radius * cos_a[i], outer_radius * sin_a[i], z_top],
                      [outer_radius * cos_a[i + 1], outer_radius * sin_a[i + 1], z_top]]
        idx += 1
        faces[idx] = [[inner_radius * cos_a[i], inner_radius * sin_a[i], z_top],
                      [outer_radius * cos_a[i + 1], outer_radius * sin_a[i + 1], z_top],
                      [inner_radius * cos_a[i + 1], inner_radius * sin_a[i + 1], z_top]]
        idx += 1

    # Bottom surface
    for i in range(n):
        faces[idx] = [[inner_radius * cos_a[i], inner_radius * sin_a[i], z_bottom],
                      [outer_radius * cos_a[i + 1], outer_radius * sin_a[i + 1], z_bottom],
                      [outer_radius * cos_a[i], outer_radius * sin_a[i], z_bottom]]
        idx += 1
        faces[idx] = [[inner_radius * cos_a[i], inner_radius * sin_a[i], z_bottom],
                      [inner_radius * cos_a[i + 1], inner_radius * sin_a[i + 1], z_bottom],
                      [outer_radius * cos_a[i + 1], outer_radius * sin_a[i + 1], z_bottom]]
        idx += 1

    # Left end cap
    faces[idx] = [[inner_radius * cos_a[0], inner_radius * sin_a[0], z_bottom],
                  [outer_radius * cos_a[0], outer_radius * sin_a[0], z_bottom],
                  [outer_radius * cos_a[0], outer_radius * sin_a[0], z_top]]
    idx += 1
    faces[idx] = [[inner_radius * cos_a[0], inner_radius * sin_a[0], z_bottom],
                  [outer_radius * cos_a[0], outer_radius * sin_a[0], z_top],
                  [inner_radius * cos_a[0], inner_radius * sin_a[0], z_top]]
    idx += 1

    # Right end cap
    faces[idx] = [[inner_radius * cos_a[-1], inner_radius * sin_a[-1], z_bottom],
                  [outer_radius * cos_a[-1], outer_radius * sin_a[-1], z_top],
                  [outer_radius * cos_a[-1], outer_radius * sin_a[-1], z_bottom]]
    idx += 1
    faces[idx] = [[inner_radius * cos_a[-1], inner_radius * sin_a[-1], z_bottom],
                  [inner_radius * cos_a[-1], inner_radius * sin_a[-1], z_top],
                  [outer_radius * cos_a[-1], outer_radius * sin_a[-1], z_top]]
    idx += 1

    return faces[:idx]
