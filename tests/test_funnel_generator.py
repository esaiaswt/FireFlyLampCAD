"""Property-based tests for funnel_generator module.

Uses Hypothesis to verify geometric invariants of generated meshes.
"""

import sys
import os
import numpy as np
import trimesh
from hypothesis import given, settings
from hypothesis.strategies import floats, integers, composite

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parameter_validator import FunnelParams
from funnel_generator import generate_funnel_mesh, generate_cylinder_sleeve_mesh, assemble_model


def to_trimesh(stl_mesh_obj):
    """Convert a numpy-stl Mesh to a trimesh Trimesh for analysis.

    STL format stores vertices per-face (no shared indices), so we must
    merge coincident vertices for trimesh to correctly detect shared edges
    and evaluate watertightness.
    """
    vectors = stl_mesh_obj.vectors.reshape(-1, 3)
    faces = np.arange(len(vectors)).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vectors, faces=faces, process=True)
    return mesh


@composite
def valid_funnel_params(draw):
    """Strategy that generates valid FunnelParams for cylinder sleeve testing."""
    bottom_d = draw(floats(min_value=5.0, max_value=200.0, allow_nan=False, allow_infinity=False))
    top_d = draw(floats(min_value=5.0, max_value=200.0, allow_nan=False, allow_infinity=False))
    height = draw(floats(min_value=1.0, max_value=200.0, allow_nan=False, allow_infinity=False))
    min_d = min(bottom_d, top_d)
    wall = draw(floats(min_value=0.1, max_value=min_d / 2 - 0.01, allow_nan=False, allow_infinity=False))
    sleeve_h = draw(floats(min_value=1.0, max_value=200.0, allow_nan=False, allow_infinity=False))
    sleeve_wall = draw(floats(min_value=0.1, max_value=top_d / 2 - 0.01, allow_nan=False, allow_infinity=False))
    segments = draw(integers(min_value=3, max_value=128))
    return FunnelParams(
        bottom_diameter=bottom_d,
        top_diameter=top_d,
        funnel_height=height,
        wall_thickness=wall,
        sleeve_height=sleeve_h,
        sleeve_wall_thickness=sleeve_wall,
        num_segments=segments,
    )


# Feature: funnel-stl-viewer, Property 6: Cylinder sleeve positioning
# Validates: Requirements 2.2
class TestCylinderSleevePositioning:
    """Property 6: Cylinder sleeve positioning.

    For any valid cylinder sleeve parameters, the top edge of the generated
    sleeve mesh SHALL have a Z coordinate equal to the funnel height (brim
    alignment), and the bottom edge Z coordinate SHALL equal
    funnel_height - sleeve_height.

    **Validates: Requirements 2.2**
    """

    @given(params=valid_funnel_params())
    @settings(max_examples=100)
    def test_cylinder_sleeve_z_positioning(self, params: FunnelParams):
        """Top edge Z == funnel_height, bottom edge Z == funnel_height - sleeve_height."""
        sleeve_mesh = generate_cylinder_sleeve_mesh(params)

        # Extract all Z coordinates from mesh vertices
        # numpy-stl stores vertices in mesh.vectors with shape (n_faces, 3, 3)
        # vectors[i][j] is the j-th vertex of the i-th face, each vertex is [x, y, z]
        all_z = sleeve_mesh.vectors[:, :, 2].flatten()

        expected_z_top = params.funnel_height
        expected_z_bottom = params.funnel_height - params.sleeve_height

        actual_z_max = np.max(all_z)
        actual_z_min = np.min(all_z)

        assert abs(actual_z_max - expected_z_top) < 1e-6, (
            f"Top edge Z mismatch: expected {expected_z_top}, got {actual_z_max}"
        )
        assert abs(actual_z_min - expected_z_bottom) < 1e-6, (
            f"Bottom edge Z mismatch: expected {expected_z_bottom}, got {actual_z_min}"
        )


# Feature: funnel-stl-viewer, Property 2: Funnel mesh is watertight
# Validates: Requirements 1.3
class TestFunnelMeshWatertight:
    """Property 2: Funnel mesh is watertight.

    For any valid set of funnel parameters (positive dimensions,
    wall thickness < half smallest diameter, segments >= 3),
    the generated funnel mesh SHALL be watertight with zero
    non-manifold edges.

    **Validates: Requirements 1.3**
    """

    @given(params=valid_funnel_params())
    @settings(max_examples=100, deadline=None)
    def test_funnel_mesh_is_watertight(self, params: FunnelParams):
        """Generated funnel mesh is watertight for all valid params."""
        funnel = generate_funnel_mesh(params)
        tri_mesh = to_trimesh(funnel)

        assert tri_mesh.is_watertight, (
            f"Funnel mesh is not watertight for params: "
            f"bottom_d={params.bottom_diameter}, top_d={params.top_diameter}, "
            f"height={params.funnel_height}, wall={params.wall_thickness}, "
            f"segments={params.num_segments}"
        )


@composite
def valid_sleeve_params(draw):
    """Strategy that generates valid FunnelParams focused on cylinder sleeve testing.

    Ranges:
    - top_d in [5.0, 200.0]
    - sleeve_h in [1.0, 200.0]
    - sleeve_wall in [0.1, top_d/2 - 0.01]
    - segments in [3, 128]
    - Other params use defaults
    """
    top_d = draw(floats(min_value=5.0, max_value=200.0, allow_nan=False, allow_infinity=False))
    sleeve_h = draw(floats(min_value=1.0, max_value=200.0, allow_nan=False, allow_infinity=False))
    max_sleeve_wall = top_d / 2.0 - 0.01
    sleeve_wall = draw(floats(min_value=0.1, max_value=max_sleeve_wall, allow_nan=False, allow_infinity=False))
    segments = draw(integers(min_value=3, max_value=128))

    return FunnelParams(
        bottom_diameter=52.0,
        top_diameter=top_d,
        funnel_height=30.0,
        wall_thickness=1.2,
        sleeve_height=sleeve_h,
        sleeve_wall_thickness=sleeve_wall,
        num_segments=segments,
    )


# Feature: funnel-stl-viewer, Property 3: Cylinder sleeve mesh is watertight
# Validates: Requirements 2.3
class TestCylinderSleeveWatertight:
    """Property 3: Cylinder sleeve mesh is watertight.

    For any valid set of cylinder sleeve parameters (positive height,
    positive wall thickness, segments >= 3), the generated cylinder sleeve
    mesh SHALL be watertight with zero non-manifold edges.

    **Validates: Requirements 2.3**
    """

    @given(params=valid_sleeve_params())
    @settings(max_examples=100)
    def test_cylinder_sleeve_watertight(self, params: FunnelParams):
        """Generated cylinder sleeve mesh is watertight for all valid params."""
        from collections import Counter

        sleeve_mesh = generate_cylinder_sleeve_mesh(params)
        tri_mesh = to_trimesh(sleeve_mesh)

        # Merge duplicate vertices for proper topology analysis
        tri_mesh.merge_vertices()

        assert tri_mesh.is_watertight, (
            f"Cylinder sleeve mesh is not watertight for params: "
            f"top_diameter={params.top_diameter}, sleeve_height={params.sleeve_height}, "
            f"sleeve_wall_thickness={params.sleeve_wall_thickness}, "
            f"num_segments={params.num_segments}"
        )

        # Check for zero non-manifold edges
        edges = tri_mesh.edges_sorted
        edge_counts = Counter(map(tuple, edges))
        non_manifold_edges = sum(1 for count in edge_counts.values() if count > 2)

        assert non_manifold_edges == 0, (
            f"Cylinder sleeve mesh has {non_manifold_edges} non-manifold edges for params: "
            f"top_diameter={params.top_diameter}, sleeve_height={params.sleeve_height}, "
            f"sleeve_wall_thickness={params.sleeve_wall_thickness}, "
            f"num_segments={params.num_segments}"
        )


# Feature: funnel-stl-viewer, Property 5: Inner diameter derivation
# Validates: Requirements 1.2
class TestInnerDiameterDerivation:
    """Property 5: Inner diameter derivation.

    For any valid funnel parameters, the computed inner diameters SHALL equal
    the outer diameters minus twice the wall thickness:
    - bottom_inner = bottom_diameter - 2 * wall_thickness
    - top_inner = top_diameter - 2 * wall_thickness

    **Validates: Requirements 1.2**
    """

    @given(params=valid_funnel_params())
    @settings(max_examples=50)
    def test_inner_diameter_derivation(self, params: FunnelParams):
        """Inner radii at bottom (z=0) and top (z=height) match expected derivation."""
        mesh = generate_funnel_mesh(params)

        # Expected inner radii
        expected_inner_radius_bottom = (params.bottom_diameter - 2 * params.wall_thickness) / 2.0
        expected_inner_radius_top = (params.top_diameter - 2 * params.wall_thickness) / 2.0

        # Extract all vertices from the mesh faces
        # mesh.vectors has shape (num_faces, 3, 3) — each face has 3 vertices of (x,y,z)
        all_vertices = mesh.vectors.reshape(-1, 3)

        # Separate vertices by Z coordinate
        # numpy-stl uses float32 internally so we use a tolerance appropriate
        # for float32 precision (relative error ~1e-7 per coordinate)
        z_tolerance = 1e-4
        bottom_vertices = all_vertices[np.abs(all_vertices[:, 2] - 0.0) < z_tolerance]
        top_vertices = all_vertices[
            np.abs(all_vertices[:, 2] - np.float32(params.funnel_height)) < z_tolerance
        ]

        # Compute radii (distance from Z-axis) for each vertex set
        bottom_radii = np.sqrt(bottom_vertices[:, 0] ** 2 + bottom_vertices[:, 1] ** 2)
        top_radii = np.sqrt(top_vertices[:, 0] ** 2 + top_vertices[:, 1] ** 2)

        # The minimum radius at bottom should match the inner bottom radius
        # Tolerance accounts for float32 storage in numpy-stl mesh vertices
        actual_inner_radius_bottom = float(np.min(bottom_radii))
        assert abs(actual_inner_radius_bottom - expected_inner_radius_bottom) < 1e-4, (
            f"Bottom inner radius mismatch: expected {expected_inner_radius_bottom}, "
            f"got {actual_inner_radius_bottom}"
        )

        # The minimum radius at top should match the inner top radius
        actual_inner_radius_top = float(np.min(top_radii))
        assert abs(actual_inner_radius_top - expected_inner_radius_top) < 1e-4, (
            f"Top inner radius mismatch: expected {expected_inner_radius_top}, "
            f"got {actual_inner_radius_top}"
        )


# Feature: funnel-stl-viewer, Property 4: Combined model has no degenerate faces
# Validates: Requirements 1.3, 2.3, 3.2
class TestNoDegenerateFaces:
    """Property 4: Combined model has no degenerate faces.

    For any valid combined model mesh, all triangle faces SHALL have an area
    greater than 1e-6 square millimetres.

    **Validates: Requirements 1.3, 2.3, 3.2**
    """

    @given(params=valid_funnel_params())
    @settings(max_examples=50, deadline=None)
    def test_combined_model_no_degenerate_faces(self, params: FunnelParams):
        """All faces in the combined model have area > 1e-6 mm²."""
        funnel = generate_funnel_mesh(params)
        sleeve = generate_cylinder_sleeve_mesh(params)
        combined = assemble_model(funnel, sleeve)

        tri_mesh = to_trimesh(combined)

        # Compute per-face areas
        face_areas = tri_mesh.area_faces

        # Assert all face areas are above the degenerate threshold
        min_area = np.min(face_areas)
        assert min_area > 1e-6, (
            f"Degenerate face detected: minimum face area = {min_area:.2e} mm², "
            f"expected > 1e-6 mm². Params: bottom_d={params.bottom_diameter}, "
            f"top_d={params.top_diameter}, height={params.funnel_height}, "
            f"wall={params.wall_thickness}, sleeve_h={params.sleeve_height}, "
            f"sleeve_wall={params.sleeve_wall_thickness}, segments={params.num_segments}"
        )


# Feature: funnel-stl-viewer, Property 8: Combined model face normals point outward
# Validates: Requirements 3.2
class TestCombinedModelOutwardNormals:
    """Property 8: Combined model face normals point outward.

    For any valid combined model mesh, all face normals SHALL be consistently
    oriented outward. The combined model consists of two disjoint watertight
    bodies (funnel + sleeve). Each component is verified individually by
    checking that trimesh reports it as a valid volume with positive volume
    (consistent outward normals per the right-hand rule).

    **Validates: Requirements 3.2**
    """

    @given(params=valid_funnel_params())
    @settings(max_examples=50, deadline=None)
    def test_combined_model_normals_outward(self, params: FunnelParams):
        """Each component of the combined model has consistent outward normals (positive volume)."""
        funnel = generate_funnel_mesh(params)
        sleeve = generate_cylinder_sleeve_mesh(params)
        combined = assemble_model(funnel, sleeve, params)

        # The combined model is two disjoint watertight bodies.
        # Verify each component individually: a mesh with consistently outward
        # normals will be reported as a valid volume with positive value by trimesh.
        tri_funnel = to_trimesh(funnel)
        tri_sleeve = to_trimesh(sleeve)

        # Funnel component: consistent outward normals => is_volume and volume > 0
        assert tri_funnel.is_volume, (
            f"Funnel mesh is not a valid volume (normals may be inconsistent) for params: "
            f"bottom_d={params.bottom_diameter}, top_d={params.top_diameter}, "
            f"height={params.funnel_height}, wall={params.wall_thickness}, "
            f"segments={params.num_segments}"
        )
        assert tri_funnel.volume > 0, (
            f"Funnel mesh volume is non-positive ({tri_funnel.volume}), indicating "
            f"normals are not consistently outward for params: "
            f"bottom_d={params.bottom_diameter}, top_d={params.top_diameter}, "
            f"height={params.funnel_height}, wall={params.wall_thickness}, "
            f"segments={params.num_segments}"
        )

        # Sleeve component: consistent outward normals => is_volume and volume > 0
        assert tri_sleeve.is_volume, (
            f"Sleeve mesh is not a valid volume (normals may be inconsistent) for params: "
            f"top_d={params.top_diameter}, sleeve_h={params.sleeve_height}, "
            f"sleeve_wall={params.sleeve_wall_thickness}, "
            f"segments={params.num_segments}"
        )
        assert tri_sleeve.volume > 0, (
            f"Sleeve mesh volume is non-positive ({tri_sleeve.volume}), indicating "
            f"normals are not consistently outward for params: "
            f"top_d={params.top_diameter}, sleeve_h={params.sleeve_height}, "
            f"sleeve_wall={params.sleeve_wall_thickness}, "
            f"segments={params.num_segments}"
        )
