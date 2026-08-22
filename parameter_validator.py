"""Parameter validation module for funnel STL generator.

Defines data models for funnel parameters and validation results,
plus the validate_parameters() function for checking constraints.
"""

from dataclasses import dataclass, field
from typing import List

from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FunnelParams:
    """Parametric dimensions for funnel and cylinder sleeve generation."""

    bottom_diameter: float = 52.0  # mm
    top_diameter: float = 70.0  # mm
    funnel_height: float = 30.0  # mm
    wall_thickness: float = 1.2  # mm
    sleeve_height: float = 50.0  # mm
    sleeve_wall_thickness: float = 1.2  # mm
    num_segments: int = 64  # circumferential segments


@dataclass
class ValidationResult:
    """Result of parameter validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)  # list of human-readable error messages


def validate_parameters(params: FunnelParams) -> ValidationResult:
    """
    Validates all parametric inputs for funnel and cylinder sleeve generation.

    Checks:
    - All dimensions > 0
    - All dimensions <= 500mm
    - num_segments >= 3
    - wall_thickness < min(bottom_diameter, top_diameter) / 2
    - sleeve_wall_thickness < top_diameter / 2

    Args:
        params: FunnelParams instance with user-supplied values.

    Returns:
        ValidationResult with is_valid flag and list of error strings.
    """
    logger.debug(
        "Validating parameters: bottom_diameter=%.2f, top_diameter=%.2f, "
        "funnel_height=%.2f, wall_thickness=%.2f, sleeve_height=%.2f, "
        "sleeve_wall_thickness=%.2f, num_segments=%d",
        params.bottom_diameter,
        params.top_diameter,
        params.funnel_height,
        params.wall_thickness,
        params.sleeve_height,
        params.sleeve_wall_thickness,
        params.num_segments,
    )

    errors: List[str] = []

    # Validate positive dimensions (> 0)
    dimension_fields = {
        "bottom_diameter": params.bottom_diameter,
        "top_diameter": params.top_diameter,
        "funnel_height": params.funnel_height,
        "wall_thickness": params.wall_thickness,
        "sleeve_height": params.sleeve_height,
        "sleeve_wall_thickness": params.sleeve_wall_thickness,
    }

    for field_name, value in dimension_fields.items():
        if value <= 0:
            errors.append(
                f"{field_name} must be greater than zero (got {value})"
            )

    # Validate maximum dimensions (<= 500mm)
    for field_name, value in dimension_fields.items():
        if value > 500:
            errors.append(
                f"{field_name} must not exceed 500mm (got {value})"
            )

    # Validate num_segments >= 3
    if params.num_segments < 3:
        errors.append(
            f"num_segments must be at least 3 (got {params.num_segments})"
        )

    # Validate funnel wall_thickness < min(bottom_diameter, top_diameter) / 2
    min_diameter = min(params.bottom_diameter, params.top_diameter)
    if params.wall_thickness >= min_diameter / 2:
        errors.append(
            f"wall_thickness must be less than half the smallest diameter "
            f"({min_diameter / 2:.2f}mm), got {params.wall_thickness}mm"
        )

    # Validate sleeve_wall_thickness < top_diameter / 2
    if params.sleeve_wall_thickness >= params.top_diameter / 2:
        errors.append(
            f"sleeve_wall_thickness must be less than half the top diameter "
            f"({params.top_diameter / 2:.2f}mm), got {params.sleeve_wall_thickness}mm"
        )

    is_valid = len(errors) == 0
    result = ValidationResult(is_valid=is_valid, errors=errors)

    if is_valid:
        logger.debug("Parameter validation passed")
    else:
        logger.debug("Parameter validation failed with %d error(s): %s", len(errors), errors)

    return result
