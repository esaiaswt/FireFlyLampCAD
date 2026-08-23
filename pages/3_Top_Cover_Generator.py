"""Streamlit page for Top Cover STL Generator.

Renders sidebar parameter controls for the circular top cover design with
mesh holes, coordinates the generation pipeline with progress feedback,
and provides the 3D viewer, download button, and shutdown functionality.
"""

import os
import time
import threading

import psutil
import keyboard
import streamlit as st

from top_cover_generator import generate_top_cover_mesh
from stl_exporter import export_binary_stl
from stl_viewer import render_model, render_placeholder
from logging_config import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# --- Session State Initialization ---
if "top_cover_mesh" not in st.session_state:
    st.session_state["top_cover_mesh"] = None
if "top_cover_export" not in st.session_state:
    st.session_state["top_cover_export"] = None
if "top_cover_error" not in st.session_state:
    st.session_state["top_cover_error"] = None

# --- Main Area ---
st.title("Top Cover STL Generator")
st.markdown(
    "Design a circular top cover with a diamond wire grid pattern for maximum "
    "airflow while preventing objects from passing through. "
    "Adjust parameters in the sidebar, then generate and preview your 3D model."
)

# --- Sidebar: Parameter Input Controls ---
st.sidebar.header("Cover Dimensions")

inner_diameter = st.sidebar.number_input(
    "Inner Diameter (mm)",
    min_value=10.0,
    max_value=300.0,
    value=92.0,
    step=1.0,
    format="%.1f",
    key="tc_inner_diameter",
    help="Internal diameter of the circular cover.",
)

wall_thickness = st.sidebar.number_input(
    "Wall Thickness (mm)",
    min_value=0.4,
    max_value=10.0,
    value=1.2,
    step=0.1,
    format="%.1f",
    key="tc_wall_thickness",
    help="Wall thickness of the cylindrical cover.",
)

cover_height = st.sidebar.number_input(
    "Cover Height (mm)",
    min_value=2.0,
    max_value=100.0,
    value=10.0,
    step=0.5,
    format="%.1f",
    key="tc_cover_height",
    help="Height of the cylindrical cover wall.",
)

st.sidebar.header("Mesh Hole Parameters")

grid_spacing = st.sidebar.number_input(
    "Grid Spacing (mm)",
    min_value=5.0,
    max_value=50.0,
    value=20.0,
    step=1.0,
    format="%.1f",
    key="tc_grid_spacing",
    help="Distance between parallel grid bars. Controls diamond opening size.",
)

bar_width = st.sidebar.number_input(
    "Bar Width (mm)",
    min_value=0.4,
    max_value=5.0,
    value=1.2,
    step=0.1,
    format="%.1f",
    key="tc_bar_width",
    help="Width of each grid bar.",
)

st.sidebar.header("Quality")

num_segments = st.sidebar.slider(
    "Circle Segments",
    min_value=16,
    max_value=128,
    value=64,
    step=8,
    key="tc_num_segments",
    help="Number of segments for the cylindrical wall. Higher = smoother.",
)

# --- Parameter Validation ---
errors = []

import math

if wall_thickness >= inner_diameter / 2:
    errors.append(
        f"Wall thickness ({wall_thickness}mm) must be less than "
        f"half the inner diameter ({inner_diameter / 2:.1f}mm)."
    )

if bar_width >= grid_spacing:
    errors.append(
        f"Bar width ({bar_width}mm) must be less than "
        f"grid spacing ({grid_spacing}mm)."
    )

if errors:
    for err in errors:
        st.sidebar.error(err)
    logger.debug("Top cover parameter validation failed: %s", errors)

# --- Display parameter summary ---
st.sidebar.divider()
st.sidebar.markdown("**Computed Dimensions:**")
outer_diameter = inner_diameter + 2 * wall_thickness
diamond_diagonal = grid_spacing * math.sqrt(2)
st.sidebar.markdown(f"- Outer diameter: {outer_diameter:.1f} mm")
st.sidebar.markdown(f"- Outer radius: {outer_diameter / 2:.1f} mm")
st.sidebar.markdown(f"- Inner radius: {inner_diameter / 2:.1f} mm")
st.sidebar.markdown(f"- Cover height: {cover_height:.1f} mm")
st.sidebar.markdown(f"- Diamond opening diagonal: {diamond_diagonal:.1f} mm")
st.sidebar.markdown(f"- Diamond opening side: {grid_spacing:.1f} mm")

if diamond_diagonal >= 40.0:
    st.sidebar.warning(
        f"Diamond diagonal ({diamond_diagonal:.1f}mm) exceeds 40mm. "
        "Reduce grid spacing to block 40mm objects."
    )


# --- Generation Pipeline ---
def run_top_cover_pipeline():
    """Execute the top cover generation pipeline with progress updates.

    Steps:
      1. Generate top cover mesh (cylinder wall + mesh top with holes)
      2. Export to binary STL

    Displays a progress bar with step-based updates and descriptive status
    messages.
    """
    logger.info("run_top_cover_pipeline() entry")
    pipeline_start = time.perf_counter()

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: Generate mesh
        status_text.text("Step 1 of 2: Generating top cover mesh...")
        progress_bar.progress(10)
        logger.debug("Generating top cover mesh with parameters")

        cover_mesh = generate_top_cover_mesh(
            inner_diameter=inner_diameter,
            wall_thickness=wall_thickness,
            cover_height=cover_height,
            grid_spacing=grid_spacing,
            bar_width=bar_width,
            num_segments=num_segments,
        )

        progress_bar.progress(60)
        elapsed_step1 = (time.perf_counter() - pipeline_start) * 1000
        logger.debug(f"Mesh generation complete - elapsed={elapsed_step1:.2f}ms")

        # Step 2: Export to STL
        status_text.text("Step 2 of 2: Exporting to STL...")
        progress_bar.progress(70)

        export_result = export_binary_stl(cover_mesh, filename="top_cover.stl")
        if not export_result.success:
            raise RuntimeError(f"STL export failed: {export_result.error_message}")

        progress_bar.progress(100)
        status_text.text("Top cover generation complete!")

        # Store results
        st.session_state["top_cover_mesh"] = cover_mesh
        st.session_state["top_cover_export"] = export_result
        st.session_state["top_cover_error"] = None

        total_elapsed = (time.perf_counter() - pipeline_start) * 1000
        logger.info(
            f"run_top_cover_pipeline() exit - "
            f"faces={len(cover_mesh.data)}, elapsed={total_elapsed:.2f}ms"
        )
        st.success("Top cover model generated successfully!")

    except Exception as e:
        logger.error(f"Top cover generation failed: {e}", exc_info=True)
        status_text.empty()
        st.error(f"Generation failed: {e}")
        st.session_state["top_cover_error"] = str(e)


# --- Generate Model Button ---
generate_disabled = len(errors) > 0
if st.button("Generate Top Cover", disabled=generate_disabled, type="primary"):
    with st.spinner("Generating top cover model..."):
        run_top_cover_pipeline()

# --- Display previous error if present ---
if st.session_state.get("top_cover_error"):
    st.error(f"Last generation error: {st.session_state['top_cover_error']}")

# --- 3D Viewer ---
if st.session_state.get("top_cover_mesh") is not None:
    logger.info("Attempting to render top cover 3D model")
    try:
        render_model(st.session_state["top_cover_mesh"])
        logger.info("Top cover 3D model rendered successfully")
    except Exception as e:
        logger.error("Top cover render failed: %s", e, exc_info=True)
        render_placeholder("No valid model available for rendering.")
        st.error(f"3D viewer rendering failed: {e}")
else:
    st.info("Configure parameters and click 'Generate Top Cover' to see the 3D preview.")

# --- Download Button ---
export_result = st.session_state.get("top_cover_export")
if export_result is not None and export_result.success:
    logger.info(
        "Displaying download button - file_size=%d bytes, face_count=%d",
        len(export_result.stl_bytes),
        export_result.face_count,
    )
    st.download_button(
        label="Download Top Cover STL",
        data=export_result.stl_bytes,
        file_name="top_cover.stl",
        mime="application/octet-stream",
    )
elif export_result is not None and not export_result.success:
    logger.error("Export result indicates failure: %s", export_result.error_message)
    st.error(f"STL export failed: {export_result.error_message}")


# --- Shutdown App ---
def shutdown_app():
    """Gracefully terminate the Streamlit application."""
    logger.warning("Shutdown requested by user")
    st.warning("Shutting down...")
    time.sleep(0.5)

    def force_exit():
        time.sleep(5)
        logger.error("Force-terminating after 5s deadline")
        os._exit(0)

    timer = threading.Thread(target=force_exit, daemon=True)
    timer.start()
    logger.info("5-second force-exit timer started")

    try:
        keyboard.press_and_release("ctrl+w")
        logger.info("Browser tab close shortcut sent")
        pid = os.getpid()
        logger.info(f"Terminating process PID={pid}")
        p = psutil.Process(pid)
        p.terminate()
    except Exception as e:
        logger.error(f"Shutdown failed, forcing exit: {e}")
        os._exit(0)


# --- Shutdown Button ---
st.sidebar.divider()
if st.sidebar.button("Shutdown App", type="secondary"):
    shutdown_app()
