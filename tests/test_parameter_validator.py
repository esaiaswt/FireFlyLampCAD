"""Property-based tests for parameter validation.

Property 7: Parameter validation rejects invalid inputs
Validates: Requirements 1.4, 1.5, 6.3, 6.5

For any parameter set where at least one dimension is zero or negative,
OR wall_thickness >= half the smallest diameter,
OR sleeve_wall_thickness >= top_diameter / 2,
the parameter validator SHALL return is_valid=False with a non-empty errors list.
"""

import sys
import os

# Add project root to path so imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypothesis import given, settings, assume
from hypothesis.strategies import floats, integers, composite, one_of, just, sampled_from

from parameter_validator import FunnelParams, validate_parameters, ValidationResult


# --- Strategies ---

@composite
def valid_funnel_params(draw):
    """Generate parameter sets that satisfy all validation constraints."""
    bottom_d = draw(floats(min_value=5.0, max_value=500.0))
    top_d = draw(floats(min_value=5.0, max_value=500.0))
    height = draw(floats(min_value=1.0, max_value=500.0))
    min_d = min(bottom_d, top_d)
    # wall_thickness must be < min_d / 2
    wall = draw(floats(min_value=0.1, max_value=min_d / 2 - 0.01))
    sleeve_h = draw(floats(min_value=1.0, max_value=500.0))
    # sleeve_wall_thickness must be < top_d / 2
    sleeve_wall = draw(floats(min_value=0.1, max_value=top_d / 2 - 0.01))
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


@composite
def params_with_non_positive_dimension(draw):
    """Generate params where at least one dimension is zero or negative."""
    # Start with valid-ish base values
    bottom_d = draw(floats(min_value=5.0, max_value=100.0))
    top_d = draw(floats(min_value=5.0, max_value=100.0))
    height = draw(floats(min_value=1.0, max_value=100.0))
    wall = draw(floats(min_value=0.1, max_value=2.0))
    sleeve_h = draw(floats(min_value=1.0, max_value=100.0))
    sleeve_wall = draw(floats(min_value=0.1, max_value=2.0))
    segments = draw(integers(min_value=3, max_value=128))

    # Pick which field(s) to make invalid (zero or negative)
    invalid_value = draw(floats(min_value=-100.0, max_value=0.0))
    field_to_break = draw(sampled_from([
        "bottom_diameter", "top_diameter", "funnel_height",
        "wall_thickness", "sleeve_height", "sleeve_wall_thickness"
    ]))

    values = {
        "bottom_diameter": bottom_d,
        "top_diameter": top_d,
        "funnel_height": height,
        "wall_thickness": wall,
        "sleeve_height": sleeve_h,
        "sleeve_wall_thickness": sleeve_wall,
        "num_segments": segments,
    }
    values[field_to_break] = invalid_value

    return FunnelParams(**values)


@composite
def params_with_excessive_funnel_wall_thickness(draw):
    """Generate params where wall_thickness >= min(bottom_diameter, top_diameter) / 2."""
    bottom_d = draw(floats(min_value=5.0, max_value=100.0))
    top_d = draw(floats(min_value=5.0, max_value=100.0))
    min_d = min(bottom_d, top_d)
    # wall_thickness >= min_d / 2
    wall = draw(floats(min_value=min_d / 2, max_value=min_d))
    height = draw(floats(min_value=1.0, max_value=100.0))
    sleeve_h = draw(floats(min_value=1.0, max_value=100.0))
    # Keep sleeve_wall valid to isolate the funnel wall constraint
    sleeve_wall = draw(floats(min_value=0.1, max_value=top_d / 2 - 0.01))
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


@composite
def params_with_excessive_sleeve_wall_thickness(draw):
    """Generate params where sleeve_wall_thickness >= top_diameter / 2."""
    bottom_d = draw(floats(min_value=5.0, max_value=100.0))
    top_d = draw(floats(min_value=5.0, max_value=100.0))
    min_d = min(bottom_d, top_d)
    # Keep funnel wall valid to isolate the sleeve wall constraint
    wall = draw(floats(min_value=0.1, max_value=min_d / 2 - 0.01))
    height = draw(floats(min_value=1.0, max_value=100.0))
    sleeve_h = draw(floats(min_value=1.0, max_value=100.0))
    # sleeve_wall_thickness >= top_d / 2
    sleeve_wall = draw(floats(min_value=top_d / 2, max_value=top_d))
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


# --- Property Tests ---


# Feature: funnel-stl-viewer, Property 7: Parameter validation rejects invalid inputs
# **Validates: Requirements 1.4, 1.5, 6.3, 6.5**


class TestProperty7RejectsInvalidInputs:
    """Property 7: Parameter validation rejects invalid inputs."""

    @given(params=params_with_non_positive_dimension())
    @settings(max_examples=100)
    def test_rejects_zero_or_negative_dimensions(self, params):
        """Any parameter set with a zero or negative dimension is rejected."""
        result = validate_parameters(params)
        assert result.is_valid is False, (
            f"Expected validation to fail for params with non-positive dimension, "
            f"but got is_valid=True. Params: {params}"
        )
        assert len(result.errors) > 0, (
            "Expected non-empty errors list for invalid params"
        )

    @given(params=params_with_excessive_funnel_wall_thickness())
    @settings(max_examples=100)
    def test_rejects_funnel_wall_thickness_too_large(self, params):
        """Wall thickness >= half the smallest diameter is rejected."""
        result = validate_parameters(params)
        assert result.is_valid is False, (
            f"Expected validation to fail when wall_thickness={params.wall_thickness} >= "
            f"min(bottom_d, top_d)/2={min(params.bottom_diameter, params.top_diameter)/2}, "
            f"but got is_valid=True"
        )
        assert len(result.errors) > 0, (
            "Expected non-empty errors list for excessive wall thickness"
        )

    @given(params=params_with_excessive_sleeve_wall_thickness())
    @settings(max_examples=100)
    def test_rejects_sleeve_wall_thickness_too_large(self, params):
        """Sleeve wall thickness >= half the top diameter is rejected."""
        result = validate_parameters(params)
        assert result.is_valid is False, (
            f"Expected validation to fail when sleeve_wall_thickness={params.sleeve_wall_thickness} >= "
            f"top_diameter/2={params.top_diameter/2}, "
            f"but got is_valid=True"
        )
        assert len(result.errors) > 0, (
            "Expected non-empty errors list for excessive sleeve wall thickness"
        )


class TestValidParametersPass:
    """Valid parameters always pass validation."""

    @given(params=valid_funnel_params())
    @settings(max_examples=100)
    def test_valid_params_always_pass(self, params):
        """For any valid parameter set, validation returns is_valid=True with no errors."""
        result = validate_parameters(params)
        assert result.is_valid is True, (
            f"Expected validation to pass for valid params, "
            f"but got is_valid=False with errors: {result.errors}. Params: {params}"
        )
        assert len(result.errors) == 0, (
            f"Expected empty errors list for valid params, got: {result.errors}"
        )
