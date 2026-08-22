"""Property-based tests for stl_exporter module.

Uses Hypothesis to verify STL export round-trip geometry preservation.
"""

import sys
import os

import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import floats, integers, composite

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parameter_validator import FunnelParams
from funnel_generator import generate_funnel_mesh, generate_cylinder_sleeve_mesh, assemble_model
from stl_exporter import export_binary_stl, validate_export_roundtrip


@composite
def valid_funnel_params(draw):
    """Strategy that generates valid FunnelParams for combined model testing."""
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


# Feature: funnel-stl-viewer, Property 1: STL export round-trip preserves geometry
# Validates: Requirements 3.3
class TestSTLRoundTripPreservesGeometry:
    """Property 1: STL export round-trip preserves geometry.

    For any valid combined model mesh, exporting to binary STL and
    re-importing SHALL produce a mesh with identical face count and
    vertex positions matching within a tolerance of 1e-4 mm.

    **Validates: Requirements 3.3**
    """

    @given(params=valid_funnel_params())
    @settings(max_examples=50, deadline=None)
    def test_stl_roundtrip_preserves_geometry(self, params: FunnelParams):
        """Export + re-import yields identical face count and vertex positions within 1e-4 mm."""
        # Generate funnel and sleeve meshes
        funnel = generate_funnel_mesh(params)
        sleeve = generate_cylinder_sleeve_mesh(params)

        # Assemble the combined model
        combined = assemble_model(funnel, sleeve)

        # Export to binary STL
        export_result = export_binary_stl(combined)

        assert export_result.success, (
            f"STL export failed: {export_result.error_message} for params: "
            f"bottom_d={params.bottom_diameter}, top_d={params.top_diameter}, "
            f"height={params.funnel_height}, wall={params.wall_thickness}, "
            f"segments={params.num_segments}"
        )
        assert export_result.stl_bytes is not None, "Export succeeded but stl_bytes is None"

        # Validate round-trip: re-import and compare
        roundtrip_valid = validate_export_roundtrip(
            combined, export_result.stl_bytes, tolerance=1e-4
        )

        assert roundtrip_valid, (
            f"STL round-trip validation failed for params: "
            f"bottom_d={params.bottom_diameter}, top_d={params.top_diameter}, "
            f"height={params.funnel_height}, wall={params.wall_thickness}, "
            f"sleeve_h={params.sleeve_height}, sleeve_wall={params.sleeve_wall_thickness}, "
            f"segments={params.num_segments}"
        )
