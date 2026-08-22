"""Cup holder mesh generation module.

Generates a cup holder consisting of:
- A top ring (holder ring) with specified inner diameter, wall thickness, and height
- A vertical stand on one side connecting the ring to the base leg
- A C-shaped base leg (flat plate) with the same circle shape/orientation as the top ring
- Chamfered joints between ring-stand and stand-leg for structural strength

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
    leg_arc_degrees: float = 180.0,
    leg_width: float = 10.0,
    chamfer_size: float = 3.0,
    num_segments: int = 64,
) -> stl_mesh.Mesh:
    """Generate the complete cup holder mesh.

    The cup holder has a ring at the top, a stand on one side going from
    the leg top all the way to the ring top, and a C-shaped flat plate leg
    at the bottom.

    Chamfers:
    - Inner stand wall to C-plate: triangular fillet along the inner wall
      of the stand where it meets the top of the C-plate.
    - Stand sides to ring: triangular fillets on both left and right angular
      edges of the stand where they meet the ring outer wall.

    Coordinate system:
    - The ring is centered at X=0, Y=0
    - The stand is located at the -X side of the ring (angle = pi)
    - Z=0 is the bottom of the C-leg, Z=total_height is the top of the ring
    - The stand extends from z_leg_top to z_ring_top (full height)
    - The C-leg outer radius aligns with the stand outer wall

    Args:
        ring_inner_diameter: Inner diameter of the top ring in mm.
        ring_wall_thickness: Wall thickness of the top ring in mm.
        ring_height: Height of the top ring in mm.
        stand_wall_thickness: Wall thickness of the vertical stand in mm.
        total_height: Total height from ring top to leg bottom in mm.
        leg_arc_degrees: Arc span of the C-shaped leg in degrees.
        leg_width: Width (radial thickness) of the C-shaped flat plate leg in mm.
        chamfer_size: Size of the chamfer at joints in mm.
        num_segments: Number of circumferential segments for circles.

    Returns:
        A numpy-stl Mesh object representing the complete cup holder.
    """
    start_time = time.perf_counter()
    logger.info(
        "generate_cup_holder_mesh() entry - "
        f"ring_inner_diameter={ring_inner_diameter}, "
        f"ring_wall_thickness={ring_wall_thickness}, "
        f"ring_height={ring_height}, "
        f"stand_wall_thickness={stand_wall_thickness}, "
        f"total_height={total_height}, "
        f"leg_arc_degrees={leg_arc_degrees}, "
        f"leg_width={leg_width}, "
        f"chamfer_size={chamfer_size}, "
        f"num_segments={num_segments}"
    )

    # Compute derived dimensions
    ring_inner_radius = ring_inner_diameter / 2.0
    ring_outer_radius = ring_inner_radius + ring_wall_thickness

    # Stand outer radius
    stand_outer_radius = ring_inner_radius + stand_wall_thickness

    # The C-leg outer radius aligns with the stand outer wall
    leg_outer_radius = stand_outer_radius
    leg_inner_radius = leg_outer_radius - leg_width
    leg_height = ring_wall_thickness  # thickness of the flat plate

    # Z coordinates
    z_ring_top = total_height
    z_ring_bottom = total_height - ring_height
    z_leg_top = leg_height
    z_leg_bottom = 0.0

    logger.debug(
        f"Derived dimensions: ring_inner_radius={ring_inner_radius:.2f}, "
        f"ring_outer_radius={ring_outer_radius:.2f}, "
        f"stand_outer_radius={stand_outer_radius:.2f}, "
        f"leg_inner_radius={leg_inner_radius:.2f}, "
        f"leg_outer_radius={leg_outer_radius:.2f}, "
        f"z_ring_top={z_ring_top:.2f}, z_ring_bottom={z_ring_bottom:.2f}, "
        f"z_leg_top={z_leg_top:.2f}"
    )

    # Generate individual components
    ring_faces = _generate_ring(
        ring_inner_radius, ring_outer_radius, ring_height,
        z_ring_bottom, z_ring_top, num_segments
    )

    # Stand extends from leg top all the way to ring top
    stand_faces = _generate_stand(
        ring_inner_radius, ring_outer_radius, stand_wall_thickness,
        z_leg_top, z_ring_top, num_segments
    )

    leg_faces = _generate_c_leg(
        leg_inner_radius, leg_outer_radius, leg_height,
        z_leg_bottom, z_leg_top, leg_arc_degrees, num_segments
    )

    # Chamfer 1: Inner stand wall to C-plate top
    # A triangular fillet along the stand inner wall at z_leg_top
    chamfer_bottom_faces = _generate_chamfer_stand_to_plate(
        ring_inner_radius, stand_wall_thickness,
        z_leg_top, chamfer_size, num_segments
    )

    # Chamfer 2: Both sides of stand walls to ring
    # Triangular fillets on left and right edges of the stand where it
    # meets the ring outer wall, running vertically along ring height
    chamfer_ring_faces = _generate_chamfer_stand_to_ring(
        ring_inner_radius, ring_outer_radius, stand_wall_thickness,
        z_ring_bottom, z_ring_top, chamfer_size, num_segments
    )

    # Combine all face arrays
    all_face_arrays = [
        ring_faces, stand_faces, leg_faces,
        chamfer_bottom_faces, chamfer_ring_faces,
    ]

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
    height: float,
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

    # 4 surfaces * 2 triangles per quad * n quads = 8n faces
    total_faces = 8 * n
    faces = np.zeros((total_faces, 3, 3), dtype=np.float64)
    idx = 0

    # Outer surface (normals outward)
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

    # Inner surface (normals inward - reversed winding)
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

    # Top annular ring (normals up +Z)
    for i in range(n):
        i_next = (i + 1) % n
        faces[idx] = [
            [inner_radius * cos_a[i], inner_radius * sin_a[i], z_top],
            [outer_radius * cos_a[i], outer_radius * sin_a[i], z_top],
            [outer_radius * cos_a[i_next], outer_radius * sin_a[i_next], z_top],
        ]
        idx += 1
        faces[idx] = [
            [inner_radius * cos_a[i], inner_radius * sin_a[i], z_top],
            [outer_radius * cos_a[i_next], outer_radius * sin_a[i_next], z_top],
            [inner_radius * cos_a[i_next], inner_radius * sin_a[i_next], z_top],
        ]
        idx += 1

    # Bottom annular ring (normals down -Z)
    for i in range(n):
        i_next = (i + 1) % n
        faces[idx] = [
            [inner_radius * cos_a[i], inner_radius * sin_a[i], z_bottom],
            [outer_radius * cos_a[i_next], outer_radius * sin_a[i_next], z_bottom],
            [outer_radius * cos_a[i], outer_radius * sin_a[i], z_bottom],
        ]
        idx += 1
        faces[idx] = [
            [inner_radius * cos_a[i], inner_radius * sin_a[i], z_bottom],
            [inner_radius * cos_a[i_next], inner_radius * sin_a[i_next], z_bottom],
            [outer_radius * cos_a[i_next], outer_radius * sin_a[i_next], z_bottom],
        ]
        idx += 1

    return faces


def _generate_stand(
    ring_inner_radius: float,
    ring_outer_radius: float,
    stand_wall_thickness: float,
    z_bottom: float,
    z_top: float,
    num_segments: int,
) -> np.ndarray:
    """Generate the vertical stand that connects the ring to the C-leg.

    The stand is located at the -X side of the ring (angle = pi).
    It is an arc segment of the ring's circle with the stand wall thickness,
    spanning a small angular range to provide a solid connection.

    The stand has:
    - Inner radius = ring_inner_radius
    - Outer radius = ring_inner_radius + stand_wall_thickness
    - Arc span centered at angle pi, wide enough for structural support

    Returns:
        Numpy array of shape (N, 3, 3) containing triangle vertices.
    """
    logger.debug(
        f"_generate_stand() - ring_inner_r={ring_inner_radius:.2f}, "
        f"stand_wall={stand_wall_thickness:.2f}, "
        f"z_bottom={z_bottom:.2f}, z_top={z_top:.2f}"
    )

    # Stand arc: centered at pi, spanning enough to cover the stand width
    stand_arc_half_angle = math.atan2(stand_wall_thickness * 2, ring_inner_radius)
    stand_arc_half_angle = max(stand_arc_half_angle, math.radians(15))

    # Stand radii
    inner_r = ring_inner_radius
    outer_r = ring_inner_radius + stand_wall_thickness

    # Number of segments for the stand arc
    stand_segments = max(8, int(num_segments * (2 * stand_arc_half_angle) / (2 * np.pi)))

    center_angle = math.pi
    start_angle = center_angle - stand_arc_half_angle
    end_angle = center_angle + stand_arc_half_angle

    angles = np.linspace(start_angle, end_angle, stand_segments + 1)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    n = stand_segments
    # outer/inner: 2n triangles each, top/bottom: 2n each, 2 end caps: 2 triangles each
    total_faces = 2 * n + 2 * n + 2 * n + 2 * n + 2 + 2
    faces = np.zeros((total_faces, 3, 3), dtype=np.float64)
    idx = 0

    # Outer surface
    for i in range(n):
        faces[idx] = [
            [outer_r * cos_a[i], outer_r * sin_a[i], z_bottom],
            [outer_r * cos_a[i + 1], outer_r * sin_a[i + 1], z_bottom],
            [outer_r * cos_a[i + 1], outer_r * sin_a[i + 1], z_top],
        ]
        idx += 1
        faces[idx] = [
            [outer_r * cos_a[i], outer_r * sin_a[i], z_bottom],
            [outer_r * cos_a[i + 1], outer_r * sin_a[i + 1], z_top],
            [outer_r * cos_a[i], outer_r * sin_a[i], z_top],
        ]
        idx += 1

    # Inner surface (reversed winding)
    for i in range(n):
        faces[idx] = [
            [inner_r * cos_a[i], inner_r * sin_a[i], z_bottom],
            [inner_r * cos_a[i + 1], inner_r * sin_a[i + 1], z_top],
            [inner_r * cos_a[i + 1], inner_r * sin_a[i + 1], z_bottom],
        ]
        idx += 1
        faces[idx] = [
            [inner_r * cos_a[i], inner_r * sin_a[i], z_bottom],
            [inner_r * cos_a[i], inner_r * sin_a[i], z_top],
            [inner_r * cos_a[i + 1], inner_r * sin_a[i + 1], z_top],
        ]
        idx += 1

    # Top annular surface (normals +Z)
    for i in range(n):
        faces[idx] = [
            [inner_r * cos_a[i], inner_r * sin_a[i], z_top],
            [outer_r * cos_a[i], outer_r * sin_a[i], z_top],
            [outer_r * cos_a[i + 1], outer_r * sin_a[i + 1], z_top],
        ]
        idx += 1
        faces[idx] = [
            [inner_r * cos_a[i], inner_r * sin_a[i], z_top],
            [outer_r * cos_a[i + 1], outer_r * sin_a[i + 1], z_top],
            [inner_r * cos_a[i + 1], inner_r * sin_a[i + 1], z_top],
        ]
        idx += 1

    # Bottom annular surface (normals -Z)
    for i in range(n):
        faces[idx] = [
            [inner_r * cos_a[i], inner_r * sin_a[i], z_bottom],
            [outer_r * cos_a[i + 1], outer_r * sin_a[i + 1], z_bottom],
            [outer_r * cos_a[i], outer_r * sin_a[i], z_bottom],
        ]
        idx += 1
        faces[idx] = [
            [inner_r * cos_a[i], inner_r * sin_a[i], z_bottom],
            [inner_r * cos_a[i + 1], inner_r * sin_a[i + 1], z_bottom],
            [outer_r * cos_a[i + 1], outer_r * sin_a[i + 1], z_bottom],
        ]
        idx += 1

    # Left end cap (at start_angle)
    faces[idx] = [
        [inner_r * cos_a[0], inner_r * sin_a[0], z_bottom],
        [outer_r * cos_a[0], outer_r * sin_a[0], z_bottom],
        [outer_r * cos_a[0], outer_r * sin_a[0], z_top],
    ]
    idx += 1
    faces[idx] = [
        [inner_r * cos_a[0], inner_r * sin_a[0], z_bottom],
        [outer_r * cos_a[0], outer_r * sin_a[0], z_top],
        [inner_r * cos_a[0], inner_r * sin_a[0], z_top],
    ]
    idx += 1

    # Right end cap (at end_angle)
    faces[idx] = [
        [inner_r * cos_a[-1], inner_r * sin_a[-1], z_bottom],
        [outer_r * cos_a[-1], outer_r * sin_a[-1], z_top],
        [outer_r * cos_a[-1], outer_r * sin_a[-1], z_bottom],
    ]
    idx += 1
    faces[idx] = [
        [inner_r * cos_a[-1], inner_r * sin_a[-1], z_bottom],
        [inner_r * cos_a[-1], inner_r * sin_a[-1], z_top],
        [outer_r * cos_a[-1], outer_r * sin_a[-1], z_top],
    ]
    idx += 1

    return faces[:idx]


def _generate_c_leg(
    inner_radius: float,
    outer_radius: float,
    height: float,
    z_bottom: float,
    z_top: float,
    arc_degrees: float,
    num_segments: int,
) -> np.ndarray:
    """Generate a C-shaped flat plate leg (partial ring arc).

    The C-leg has the same circle center and orientation as the top ring,
    but only spans a partial arc (default 180 degrees). The opening of the
    C faces the +X direction (away from the stand).

    The arc is centered at angle = pi (same side as stand) so the opening
    faces +X. The leg is a flat plate with radial width = outer_radius - inner_radius.

    Returns:
        Numpy array of shape (N, 3, 3) containing triangle vertices.
    """
    logger.debug(
        f"_generate_c_leg() - inner_r={inner_radius:.2f}, outer_r={outer_radius:.2f}, "
        f"height={height:.2f}, arc_degrees={arc_degrees:.1f}"
    )

    arc_rad = math.radians(arc_degrees)

    # Center the arc at angle=pi so the gap faces +X
    center_angle = math.pi
    start_angle = center_angle - arc_rad / 2.0
    end_angle = center_angle + arc_rad / 2.0

    # Number of arc segments
    arc_segments = max(8, int(num_segments * arc_rad / (2 * np.pi)))

    angles = np.linspace(start_angle, end_angle, arc_segments + 1)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    n = arc_segments
    # Outer surface: 2n, Inner surface: 2n, Top: 2n, Bottom: 2n, 2 end caps: 4
    total_faces = 8 * n + 4
    faces = np.zeros((total_faces, 3, 3), dtype=np.float64)
    idx = 0

    # Outer surface
    for i in range(n):
        faces[idx] = [
            [outer_radius * cos_a[i], outer_radius * sin_a[i], z_bottom],
            [outer_radius * cos_a[i + 1], outer_radius * sin_a[i + 1], z_bottom],
            [outer_radius * cos_a[i + 1], outer_radius * sin_a[i + 1], z_top],
        ]
        idx += 1
        faces[idx] = [
            [outer_radius * cos_a[i], outer_radius * sin_a[i], z_bottom],
            [outer_radius * cos_a[i + 1], outer_radius * sin_a[i + 1], z_top],
            [outer_radius * cos_a[i], outer_radius * sin_a[i], z_top],
        ]
        idx += 1

    # Inner surface (reversed winding)
    for i in range(n):
        faces[idx] = [
            [inner_radius * cos_a[i], inner_radius * sin_a[i], z_bottom],
            [inner_radius * cos_a[i + 1], inner_radius * sin_a[i + 1], z_top],
            [inner_radius * cos_a[i + 1], inner_radius * sin_a[i + 1], z_bottom],
        ]
        idx += 1
        faces[idx] = [
            [inner_radius * cos_a[i], inner_radius * sin_a[i], z_bottom],
            [inner_radius * cos_a[i], inner_radius * sin_a[i], z_top],
            [inner_radius * cos_a[i + 1], inner_radius * sin_a[i + 1], z_top],
        ]
        idx += 1

    # Top surface (normals +Z)
    for i in range(n):
        faces[idx] = [
            [inner_radius * cos_a[i], inner_radius * sin_a[i], z_top],
            [outer_radius * cos_a[i], outer_radius * sin_a[i], z_top],
            [outer_radius * cos_a[i + 1], outer_radius * sin_a[i + 1], z_top],
        ]
        idx += 1
        faces[idx] = [
            [inner_radius * cos_a[i], inner_radius * sin_a[i], z_top],
            [outer_radius * cos_a[i + 1], outer_radius * sin_a[i + 1], z_top],
            [inner_radius * cos_a[i + 1], inner_radius * sin_a[i + 1], z_top],
        ]
        idx += 1

    # Bottom surface (normals -Z)
    for i in range(n):
        faces[idx] = [
            [inner_radius * cos_a[i], inner_radius * sin_a[i], z_bottom],
            [outer_radius * cos_a[i + 1], outer_radius * sin_a[i + 1], z_bottom],
            [outer_radius * cos_a[i], outer_radius * sin_a[i], z_bottom],
        ]
        idx += 1
        faces[idx] = [
            [inner_radius * cos_a[i], inner_radius * sin_a[i], z_bottom],
            [inner_radius * cos_a[i + 1], inner_radius * sin_a[i + 1], z_bottom],
            [outer_radius * cos_a[i + 1], outer_radius * sin_a[i + 1], z_bottom],
        ]
        idx += 1

    # Left end cap (at start_angle)
    faces[idx] = [
        [inner_radius * cos_a[0], inner_radius * sin_a[0], z_bottom],
        [outer_radius * cos_a[0], outer_radius * sin_a[0], z_bottom],
        [outer_radius * cos_a[0], outer_radius * sin_a[0], z_top],
    ]
    idx += 1
    faces[idx] = [
        [inner_radius * cos_a[0], inner_radius * sin_a[0], z_bottom],
        [outer_radius * cos_a[0], outer_radius * sin_a[0], z_top],
        [inner_radius * cos_a[0], inner_radius * sin_a[0], z_top],
    ]
    idx += 1

    # Right end cap (at end_angle)
    faces[idx] = [
        [inner_radius * cos_a[-1], inner_radius * sin_a[-1], z_bottom],
        [outer_radius * cos_a[-1], outer_radius * sin_a[-1], z_top],
        [outer_radius * cos_a[-1], outer_radius * sin_a[-1], z_bottom],
    ]
    idx += 1
    faces[idx] = [
        [inner_radius * cos_a[-1], inner_radius * sin_a[-1], z_bottom],
        [inner_radius * cos_a[-1], inner_radius * sin_a[-1], z_top],
        [outer_radius * cos_a[-1], outer_radius * sin_a[-1], z_top],
    ]
    idx += 1

    return faces[:idx]


def _generate_chamfer_stand_to_plate(
    ring_inner_radius: float,
    stand_wall_thickness: float,
    z_leg_top: float,
    chamfer_size: float,
    num_segments: int,
) -> np.ndarray:
    """Generate a chamfer between the inner stand wall and the C-plate top.

    This is a triangular fillet running along the inner wall of the stand
    at the joint where it meets the C-plate top surface. The chamfer fills
    the 90-degree corner between the vertical inner stand wall and the
    horizontal C-plate top.

    Cross-section (looking from +Y toward -Y at center_angle=pi):
        Stand inner wall (vertical at ring_inner_radius)
        |
        |_____ C-plate top (horizontal at z_leg_top)
        |    /
        |   /  <- chamfer (45-degree angled face)
        |  /
        | /
        |/

    The chamfer extends:
    - Vertically: from z_leg_top upward by chamfer_size
    - Radially: from ring_inner_radius inward by chamfer_size

    Returns:
        Numpy array of shape (N, 3, 3) containing triangle vertices.
    """
    logger.debug(
        f"_generate_chamfer_stand_to_plate() - z_leg_top={z_leg_top:.2f}, "
        f"chamfer_size={chamfer_size:.2f}"
    )

    # Stand arc (same as _generate_stand)
    stand_arc_half_angle = math.atan2(stand_wall_thickness * 2, ring_inner_radius)
    stand_arc_half_angle = max(stand_arc_half_angle, math.radians(15))

    center_angle = math.pi
    start_angle = center_angle - stand_arc_half_angle
    end_angle = center_angle + stand_arc_half_angle

    chamfer_segments = max(4, int(num_segments * (2 * stand_arc_half_angle) / (2 * np.pi)))
    angles = np.linspace(start_angle, end_angle, chamfer_segments + 1)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    n = chamfer_segments
    inner_r = ring_inner_radius
    chamfer_inner_r = ring_inner_radius - chamfer_size  # extends inward
    z_top = z_leg_top + chamfer_size  # extends upward
    z_bottom = z_leg_top

    # The chamfer is a triangular prism along the arc with three faces:
    # 1. The angled/sloped face (hypotenuse) - visible
    # 2. The vertical face (at inner_r, from z_bottom to z_top) - against stand wall
    # 3. The horizontal face (at z_bottom, from inner_r to chamfer_inner_r) - against plate top
    # Plus 2 triangular end caps

    # Only the angled face + end caps are needed (the other two faces are
    # interior faces against existing geometry)
    # But for a proper solid, we include all three faces + end caps.

    # Angled face: 2n triangles
    # Vertical face (inner stand wall side): 2n triangles
    # Horizontal face (plate top side): 2n triangles
    # End caps: 2 triangles (one triangle each side)
    total_faces = 6 * n + 2
    faces = np.zeros((total_faces, 3, 3), dtype=np.float64)
    idx = 0

    # Angled (hypotenuse) face - the visible 45-degree surface
    # Goes from (chamfer_inner_r, z_bottom) to (inner_r, z_top)
    for i in range(n):
        p1 = [chamfer_inner_r * cos_a[i], chamfer_inner_r * sin_a[i], z_bottom]
        p2 = [chamfer_inner_r * cos_a[i + 1], chamfer_inner_r * sin_a[i + 1], z_bottom]
        p3 = [inner_r * cos_a[i + 1], inner_r * sin_a[i + 1], z_top]
        p4 = [inner_r * cos_a[i], inner_r * sin_a[i], z_top]

        # Normal pointing inward (toward center) and upward
        faces[idx] = [p1, p3, p2]
        idx += 1
        faces[idx] = [p1, p4, p3]
        idx += 1

    # Vertical face (against stand inner wall, at inner_r)
    # From z_bottom to z_top
    for i in range(n):
        p1 = [inner_r * cos_a[i], inner_r * sin_a[i], z_bottom]
        p2 = [inner_r * cos_a[i + 1], inner_r * sin_a[i + 1], z_bottom]
        p3 = [inner_r * cos_a[i + 1], inner_r * sin_a[i + 1], z_top]
        p4 = [inner_r * cos_a[i], inner_r * sin_a[i], z_top]

        # Normal pointing outward (away from center - toward stand)
        faces[idx] = [p1, p2, p3]
        idx += 1
        faces[idx] = [p1, p3, p4]
        idx += 1

    # Horizontal face (against plate top, at z_bottom)
    # From chamfer_inner_r to inner_r
    for i in range(n):
        p1 = [chamfer_inner_r * cos_a[i], chamfer_inner_r * sin_a[i], z_bottom]
        p2 = [chamfer_inner_r * cos_a[i + 1], chamfer_inner_r * sin_a[i + 1], z_bottom]
        p3 = [inner_r * cos_a[i + 1], inner_r * sin_a[i + 1], z_bottom]
        p4 = [inner_r * cos_a[i], inner_r * sin_a[i], z_bottom]

        # Normal pointing down (-Z, against plate top)
        faces[idx] = [p1, p2, p3]
        idx += 1
        faces[idx] = [p1, p3, p4]
        idx += 1

    # Left end cap (triangular, at start_angle)
    p_top = [inner_r * cos_a[0], inner_r * sin_a[0], z_top]
    p_bottom_inner = [inner_r * cos_a[0], inner_r * sin_a[0], z_bottom]
    p_bottom_outer = [chamfer_inner_r * cos_a[0], chamfer_inner_r * sin_a[0], z_bottom]
    faces[idx] = [p_top, p_bottom_outer, p_bottom_inner]
    idx += 1

    # Right end cap (triangular, at end_angle)
    p_top = [inner_r * cos_a[-1], inner_r * sin_a[-1], z_top]
    p_bottom_inner = [inner_r * cos_a[-1], inner_r * sin_a[-1], z_bottom]
    p_bottom_outer = [chamfer_inner_r * cos_a[-1], chamfer_inner_r * sin_a[-1], z_bottom]
    faces[idx] = [p_top, p_bottom_inner, p_bottom_outer]
    idx += 1

    return faces[:idx]


def _generate_chamfer_stand_to_ring(
    ring_inner_radius: float,
    ring_outer_radius: float,
    stand_wall_thickness: float,
    z_ring_bottom: float,
    z_ring_top: float,
    chamfer_size: float,
    num_segments: int,
) -> np.ndarray:
    """Generate chamfers on both sides of the stand where it meets the ring.

    These are vertical triangular fillets at the left and right angular edges
    of the stand, bridging the gap between the stand's side edges and the
    ring outer wall. They run along the full ring height.

    The stand is thicker (3mm) than the ring wall (1.2mm), so the stand
    protrudes outward beyond the ring. The chamfer connects the ring outer
    surface to the stand side edge, creating a smooth structural transition.

    Looking from above (top view), at each side edge of the stand:
        Ring outer wall (radius = ring_outer_radius)
        |
        |----Stand edge (radius up to stand_outer_radius)
        |   /
        |  /  <- chamfer (angled face bridging the two)
        | /
        |/

    The chamfer is a triangular prism running vertically from z_ring_bottom
    to z_ring_top. At each side:
    - One edge is on the ring outer wall (at ring_outer_radius)
    - The other edge is on the stand side face
    - The chamfer spans chamfer_size along the ring circumference from the
      stand edge, tapering from stand_outer_radius down to ring_outer_radius.

    Returns:
        Numpy array of shape (N, 3, 3) containing triangle vertices.
    """
    logger.debug(
        f"_generate_chamfer_stand_to_ring() - z_ring_bottom={z_ring_bottom:.2f}, "
        f"z_ring_top={z_ring_top:.2f}, chamfer_size={chamfer_size:.2f}"
    )

    stand_arc_half_angle = math.atan2(stand_wall_thickness * 2, ring_inner_radius)
    stand_arc_half_angle = max(stand_arc_half_angle, math.radians(15))

    center_angle = math.pi
    stand_outer_r = ring_inner_radius + stand_wall_thickness
    ring_outer_r = ring_outer_radius

    # Chamfer angular extent along the ring from the stand edge
    chamfer_angle = chamfer_size / ring_outer_r

    faces_list = []

    for side in ['left', 'right']:
        if side == 'left':
            # Left edge of stand is at (center_angle - stand_arc_half_angle)
            # Chamfer extends from stand edge in negative angle direction
            stand_edge_angle = center_angle - stand_arc_half_angle
            chamfer_end_angle = stand_edge_angle - chamfer_angle
            # At stand_edge_angle: radius = stand_outer_r
            # At chamfer_end_angle: radius = ring_outer_r
            angles = np.linspace(chamfer_end_angle, stand_edge_angle, 5)
        else:
            # Right edge of stand is at (center_angle + stand_arc_half_angle)
            # Chamfer extends from stand edge in positive angle direction
            stand_edge_angle = center_angle + stand_arc_half_angle
            chamfer_end_angle = stand_edge_angle + chamfer_angle
            # At stand_edge_angle: radius = stand_outer_r
            # At chamfer_end_angle: radius = ring_outer_r
            angles = np.linspace(stand_edge_angle, chamfer_end_angle, 5)

        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        n = len(angles) - 1

        # For each segment along the chamfer angle, interpolate the radius
        # from stand_outer_r at the stand edge to ring_outer_r at the far end
        for i in range(n):
            if side == 'left':
                # angles go from chamfer_end to stand_edge
                # t=0 at chamfer_end (ring_outer_r), t=1 at stand_edge (stand_outer_r)
                t0 = float(i) / float(n)
                t1 = float(i + 1) / float(n)
            else:
                # angles go from stand_edge to chamfer_end
                # t=0 at stand_edge (stand_outer_r), t=1 at chamfer_end (ring_outer_r)
                t0 = 1.0 - float(i) / float(n)
                t1 = 1.0 - float(i + 1) / float(n)

            r0 = ring_outer_r + (stand_outer_r - ring_outer_r) * t0
            r1 = ring_outer_r + (stand_outer_r - ring_outer_r) * t1

            # Four vertices of this segment (vertical quad from z_ring_bottom to z_ring_top)
            p_bot_0 = [r0 * cos_a[i], r0 * sin_a[i], z_ring_bottom]
            p_bot_1 = [r1 * cos_a[i + 1], r1 * sin_a[i + 1], z_ring_bottom]
            p_top_0 = [r0 * cos_a[i], r0 * sin_a[i], z_ring_top]
            p_top_1 = [r1 * cos_a[i + 1], r1 * sin_a[i + 1], z_ring_top]

            # Outer face (the angled/sloped visible surface)
            faces_list.append([p_bot_0, p_bot_1, p_top_1])
            faces_list.append([p_bot_0, p_top_1, p_top_0])

        # Top cap (horizontal at z_ring_top)
        # Triangle from ring_outer_r at far end to stand_outer_r at stand edge
        for i in range(n):
            if side == 'left':
                t0 = float(i) / float(n)
                t1 = float(i + 1) / float(n)
            else:
                t0 = 1.0 - float(i) / float(n)
                t1 = 1.0 - float(i + 1) / float(n)

            r0 = ring_outer_r + (stand_outer_r - ring_outer_r) * t0
            r1 = ring_outer_r + (stand_outer_r - ring_outer_r) * t1

            p0 = [r0 * cos_a[i], r0 * sin_a[i], z_ring_top]
            p1 = [r1 * cos_a[i + 1], r1 * sin_a[i + 1], z_ring_top]
            p_ring = [ring_outer_r * cos_a[i], ring_outer_r * sin_a[i], z_ring_top]
            p_ring_next = [ring_outer_r * cos_a[i + 1], ring_outer_r * sin_a[i + 1], z_ring_top]

            # Top face triangles (normals up)
            faces_list.append([p_ring, p0, p1])
            faces_list.append([p_ring, p1, p_ring_next])

        # Bottom cap (horizontal at z_ring_bottom)
        for i in range(n):
            if side == 'left':
                t0 = float(i) / float(n)
                t1 = float(i + 1) / float(n)
            else:
                t0 = 1.0 - float(i) / float(n)
                t1 = 1.0 - float(i + 1) / float(n)

            r0 = ring_outer_r + (stand_outer_r - ring_outer_r) * t0
            r1 = ring_outer_r + (stand_outer_r - ring_outer_r) * t1

            p0 = [r0 * cos_a[i], r0 * sin_a[i], z_ring_bottom]
            p1 = [r1 * cos_a[i + 1], r1 * sin_a[i + 1], z_ring_bottom]
            p_ring = [ring_outer_r * cos_a[i], ring_outer_r * sin_a[i], z_ring_bottom]
            p_ring_next = [ring_outer_r * cos_a[i + 1], ring_outer_r * sin_a[i + 1], z_ring_bottom]

            # Bottom face triangles (normals down)
            faces_list.append([p_ring, p1, p0])
            faces_list.append([p_ring, p_ring_next, p1])

        # Inner face (at ring_outer_r between the chamfer arc span) - vertical wall
        for i in range(n):
            p_bot = [ring_outer_r * cos_a[i], ring_outer_r * sin_a[i], z_ring_bottom]
            p_bot_next = [ring_outer_r * cos_a[i + 1], ring_outer_r * sin_a[i + 1], z_ring_bottom]
            p_top = [ring_outer_r * cos_a[i], ring_outer_r * sin_a[i], z_ring_top]
            p_top_next = [ring_outer_r * cos_a[i + 1], ring_outer_r * sin_a[i + 1], z_ring_top]

            # Normal pointing inward (toward center)
            faces_list.append([p_bot, p_top_next, p_bot_next])
            faces_list.append([p_bot, p_top, p_top_next])

    if not faces_list:
        return np.zeros((0, 3, 3), dtype=np.float64)

    faces = np.array(faces_list, dtype=np.float64)
    return faces
