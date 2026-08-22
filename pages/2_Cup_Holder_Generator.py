"""Streamlit page for Cup Holder STL Generator.

Renders sidebar parameter controls for the cup holder design, coordinates
the generation pipeline with progress feedback, and provides the 3D viewer,
download button, and shutdown functionality.
"""

import os
import time
import threading

import psutil
import keyboard
import streamlit as st

from cup_holder_generator import generate_cup_holder_mesh
from stl_exporter import export_binary_stl
from stl_viewer import render_model, render_placeholder
from logging_config import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# --- Session State Initialization ---
if "cup_holder_mesh" not in st.session_state:
    st.session_state["cup_holder_mesh"] = None
if "cup_holder_export" not in st.session_state:
    st.session_state["cup_holder_export"] = None
if "cup_holder_error" not in st.session_state:
    st.session_state["cup_holder_error"] = None

# --- Main Area ---
st.title("Cup Holder STL Generator")
st.markdown(
    "Design a cup holder with a top ring, vertical stand, and C-shaped base leg. "
    "Adjust parameters in the sidebar, then generate and preview your 3D model."
)

# --- Sidebar: Parameter Input Controls ---
st.sidebar.header("Ring Parameters")

ring_inner_diameter = st.sidebar.number_input(
    "Ring Inner Diameter (mm)",
    min_value=10.0,
    max_value=200.0,
    value=62.0,
    step=0.5,
    format="%.1f",
    key="ch_ring_inner_diameter",
    help="Inner diameter of the top holding ring.",
)

ring_wall_thickness = st.sidebar.number_input(
    "Ring Wall Thickness (mm)",
    min_value=0.4,
    max_value=10.0,
    value=1.2,
    step=0.1,
    format="%.1f",
    key="ch_ring_wall_thickness",
    help="Wall thickness of the top ring.",
)

ring_height = st.sidebar.number_input(
    "Ring Height (mm)",
    min_value=5.0,
    max_value=100.0,
    value=15.0,
    step=0.5,
    format="%.1f",
    key="ch_ring_height",
    help="Height of the top ring wall.",
)

st.sidebar.header("Stand Parameters")

total_height = st.sidebar.number_input(
    "Total Height - Ring Top to Leg Bottom (mm)",
    min_value=20.0,
    max_value=200.0,
    value=40.0,
    step=1.0,
    format="%.1f",
    key="ch_total_height",
    help="Total height from top of ring to bottom of the C-leg.",
)

stand_wall_thickness = st.sidebar.number_input(
    "Stand Wall Thickness (mm)",
    min_value=1.0,
    max_value=20.0,
    value=3.0,
    step=0.5,
    format="%.1f",
    key="ch_stand_wall_thickness",
    help="Wall thickness of the vertical stand connecting ring to leg.",
)

st.sidebar.header("C-Leg Parameters")

leg_width = st.sidebar.number_input(
    "Leg Width / Radial Thickness (mm)",
    min_value=3.0,
    max_value=50.0,
    value=10.0,
    step=1.0,
    format="%.1f",
    key="ch_leg_width",
    help="Radial width of the C-shaped flat plate leg.",
)

leg_arc_degrees = st.sidebar.slider(
    "Leg Arc Span (degrees)",
    min_value=90,
    max_value=330,
    value=180,
    step=10,
    key="ch_leg_arc_degrees",
    help="Angular span of the C-shaped base leg. 360 would be a full ring.",
)

st.sidebar.header("Joint Parameters")

chamfer_size = st.sidebar.number_input(
    "Chamfer Size (mm)",
    min_value=0.5,
    max_value=10.0,
    value=3.0,
    step=0.5,
    format="%.1f",
    key="ch_chamfer_size",
    help="Size of the reinforcing chamfers at stand-to-plate and stand-to-ring joints.",
)

st.sidebar.header("Quality")

num_segments = st.sidebar.slider(
    "Circle Segments",
    min_value=16,
    max_value=128,
    value=64,
    step=8,
    key="ch_num_segments",
    help="Number of segments for circular geometry. Higher = smoother.",
)

# --- Parameter Validation ---
errors = []

if total_height <= ring_height:
    errors.append(
        f"Total height ({total_height}mm) must be greater than ring height "
        f"({ring_height}mm) to allow room for the stand."
    )

if ring_wall_thickness >= ring_inner_diameter / 2:
    errors.append(
        f"Ring wall thickness ({ring_wall_thickness}mm) must be less than "
        f"half the inner diameter ({ring_inner_diameter / 2:.1f}mm)."
    )

if errors:
    for err in errors:
        st.sidebar.error(err)
    logger.debug("Cup holder parameter validation failed: %s", errors)

# --- Display parameter summary ---
st.sidebar.divider()
st.sidebar.markdown("**Computed Dimensions:**")
st.sidebar.markdown(f"- Ring outer diameter: {ring_inner_diameter + 2 * ring_wall_thickness:.1f} mm")
st.sidebar.markdown(f"- Stand height: {total_height - ring_height:.1f} mm")
st.sidebar.markdown(f"- Leg outer diameter: {ring_inner_diameter + 2 * leg_width:.1f} mm")
st.sidebar.markdown(f"- Leg arc: {leg_arc_degrees}\u00b0 of circle")


# --- Generation Pipeline ---
def run_cup_holder_pipeline():
    """Execute the cup holder generation pipeline with progress updates.

    Steps:
      1. Generate cup holder mesh (ring + stand + leg + chamfers)
      2. Export to binary STL

    Displays a progress bar with step-based updates and descriptive status
    messages.
    """
    logger.info("run_cup_holder_pipeline() entry")
    pipeline_start = time.perf_counter()

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: Generate mesh
        status_text.text("Step 1 of 2: Generating cup holder mesh...")
        progress_bar.progress(10)
        logger.debug("Generating cup holder mesh with parameters")

        cup_mesh = generate_cup_holder_mesh(
            ring_inner_diameter=ring_inner_diameter,
            ring_wall_thickness=ring_wall_thickness,
            ring_height=ring_height,
            stand_wall_thickness=stand_wall_thickness,
            total_height=total_height,
            leg_arc_degrees=float(leg_arc_degrees),
            leg_width=leg_width,
            chamfer_size=chamfer_size,
            num_segments=num_segments,
        )

        progress_bar.progress(60)
        elapsed_step1 = (time.perf_counter() - pipeline_start) * 1000
        logger.debug(f"Mesh generation complete - elapsed={elapsed_step1:.2f}ms")

        # Step 2: Export to STL
        status_text.text("Step 2 of 2: Exporting to STL...")
        progress_bar.progress(70)

        export_result = export_binary_stl(cup_mesh, filename="cup_holder.stl")
        if not export_result.success:
            raise RuntimeError(f"STL export failed: {export_result.error_message}")

        progress_bar.progress(100)
        status_text.text("Cup holder generation complete!")

        # Store results
        st.session_state["cup_holder_mesh"] = cup_mesh
        st.session_state["cup_holder_export"] = export_result
        st.session_state["cup_holder_error"] = None

        total_elapsed = (time.perf_counter() - pipeline_start) * 1000
        logger.info(
            f"run_cup_holder_pipeline() exit - "
            f"faces={len(cup_mesh.data)}, elapsed={total_elapsed:.2f}ms"
        )
        st.success("Cup holder model generated successfully!")

    except Exception as e:
        logger.error(f"Cup holder generation failed: {e}", exc_info=True)
        status_text.empty()
        st.error(f"Generation failed: {e}")
        st.session_state["cup_holder_error"] = str(e)


# --- Generate Model Button ---
generate_disabled = len(errors) > 0
if st.button("Generate Cup Holder", disabled=generate_disabled, type="primary"):
    with st.spinner("Generating cup holder model..."):
        run_cup_holder_pipeline()

# --- Display previous error if present ---
if st.session_state.get("cup_holder_error"):
    st.error(f"Last generation error: {st.session_state['cup_holder_error']}")

# --- 3D Viewer ---
if st.session_state.get("cup_holder_mesh") is not None:
    logger.info("Attempting to render cup holder 3D model")
    try:
        render_model(st.session_state["cup_holder_mesh"])
        logger.info("Cup holder 3D model rendered successfully")
    except Exception as e:
        logger.error("Cup holder render failed: %s", e, exc_info=True)
        render_placeholder("No valid model available for rendering.")
        st.error(f"3D viewer rendering failed: {e}")
else:
    st.info("Configure parameters and click 'Generate Cup Holder' to see the 3D preview.")

# --- Download Button ---
export_result = st.session_state.get("cup_holder_export")
if export_result is not None and export_result.success:
    logger.info(
        "Displaying download button - file_size=%d bytes, face_count=%d",
        len(export_result.stl_bytes),
        export_result.face_count,
    )
    st.download_button(
        label="Download Cup Holder STL",
        data=export_result.stl_bytes,
        file_name="cup_holder.stl",
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
