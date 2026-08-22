"""Streamlit page for Funnel STL Generator.

Renders sidebar parameter controls with inline validation, coordinates the
generation pipeline with progress feedback, and provides the 3D viewer,
download button, and shutdown functionality.
"""

import os
import time
import threading

import psutil
import keyboard
import streamlit as st

from parameter_validator import FunnelParams, validate_parameters
from funnel_generator import generate_funnel_mesh, generate_cylinder_sleeve_mesh, assemble_model
from mesh_validator import validate_mesh
from stl_exporter import export_binary_stl
from stl_viewer import render_model, render_placeholder
from logging_config import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# --- Session State Initialization ---
if "funnel_params" not in st.session_state:
    st.session_state["funnel_params"] = FunnelParams()
    logger.info("Session state initialized with default FunnelParams")
if "combined_mesh" not in st.session_state:
    st.session_state["combined_mesh"] = None
if "export_result" not in st.session_state:
    st.session_state["export_result"] = None
if "generation_error" not in st.session_state:
    st.session_state["generation_error"] = None

# --- Sidebar: Parameter Input Controls ---
st.sidebar.header("Funnel Parameters")

bottom_diameter = st.sidebar.number_input(
    "Funnel Bottom Diameter (mm)",
    min_value=0.1,
    max_value=500.0,
    value=st.session_state["funnel_params"].bottom_diameter,
    step=0.1,
    format="%.1f",
    key="input_bottom_diameter",
)

top_diameter = st.sidebar.number_input(
    "Funnel Top Diameter (mm)",
    min_value=0.1,
    max_value=500.0,
    value=st.session_state["funnel_params"].top_diameter,
    step=0.1,
    format="%.1f",
    key="input_top_diameter",
)

funnel_height = st.sidebar.number_input(
    "Funnel Height (mm)",
    min_value=0.1,
    max_value=500.0,
    value=st.session_state["funnel_params"].funnel_height,
    step=0.1,
    format="%.1f",
    key="input_funnel_height",
)

wall_thickness = st.sidebar.number_input(
    "Funnel Wall Thickness (mm)",
    min_value=0.1,
    max_value=250.0,
    value=st.session_state["funnel_params"].wall_thickness,
    step=0.1,
    format="%.1f",
    key="input_wall_thickness",
)

st.sidebar.header("Cylinder Sleeve Parameters")

sleeve_height = st.sidebar.number_input(
    "Cylinder Sleeve Height (mm)",
    min_value=0.1,
    max_value=500.0,
    value=st.session_state["funnel_params"].sleeve_height,
    step=0.1,
    format="%.1f",
    key="input_sleeve_height",
)

sleeve_wall_thickness = st.sidebar.number_input(
    "Cylinder Sleeve Wall Thickness (mm)",
    min_value=0.1,
    max_value=250.0,
    value=st.session_state["funnel_params"].sleeve_wall_thickness,
    step=0.1,
    format="%.1f",
    key="input_sleeve_wall_thickness",
)

# --- Construct FunnelParams from user inputs ---
params = FunnelParams(
    bottom_diameter=bottom_diameter,
    top_diameter=top_diameter,
    funnel_height=funnel_height,
    wall_thickness=wall_thickness,
    sleeve_height=sleeve_height,
    sleeve_wall_thickness=sleeve_wall_thickness,
)

# --- Validate parameters ---
validation_result = validate_parameters(params)

if not validation_result.is_valid:
    for error_msg in validation_result.errors:
        st.sidebar.error(error_msg)
    logger.debug(
        "Parameter validation failed: %s", validation_result.errors
    )
else:
    # Store valid params in session state
    st.session_state["funnel_params"] = params
    logger.debug("Parameters updated in session state: %s", params)

# --- Main Area ---
st.title("Funnel STL Generator")
st.markdown(
    "Adjust funnel and cylinder sleeve parameters in the sidebar, "
    "then generate and preview your 3D model."
)


# --- Generation Pipeline ---
def run_generation_pipeline(params: FunnelParams):
    """Execute the 4-step model generation pipeline with progress updates.

    Steps:
      1. Generate funnel mesh
      2. Generate cylinder sleeve mesh
      3. Assemble combined model
      4. Export to binary STL

    Displays a progress bar with step-based updates (25% per step) and
    descriptive status messages. On failure, freezes the progress bar and
    displays an error identifying the failed step. On success, stores results
    in session state.
    """
    logger.info("run_generation_pipeline() entry")
    pipeline_start = time.perf_counter()

    progress_bar = st.progress(0)
    status_text = st.empty()

    steps = [
        ("Step 1 of 4: Generating funnel mesh...", "generate_funnel"),
        ("Step 2 of 4: Generating cylinder mesh...", "generate_sleeve"),
        ("Step 3 of 4: Assembling model...", "assemble"),
        ("Step 4 of 4: Exporting to STL...", "export"),
    ]

    combined_mesh = None
    export_result = None

    for step_index, (step_label, step_key) in enumerate(steps):
        step_num = step_index + 1
        progress_pct = int((step_index / 4) * 100)
        progress_bar.progress(progress_pct)
        status_text.text(step_label)
        logger.debug("Pipeline step %d entry: %s", step_num, step_key)
        step_start = time.perf_counter()

        try:
            if step_key == "generate_funnel":
                funnel_mesh = generate_funnel_mesh(params)
            elif step_key == "generate_sleeve":
                sleeve_mesh = generate_cylinder_sleeve_mesh(params)
            elif step_key == "assemble":
                combined_mesh = assemble_model(funnel_mesh, sleeve_mesh, params)
                # Validate the combined mesh
                validation = validate_mesh(combined_mesh)
                if validation.errors:
                    logger.warning(
                        "Mesh validation warnings: %s", validation.errors
                    )
            elif step_key == "export":
                export_result = export_binary_stl(combined_mesh)
                if not export_result.success:
                    raise RuntimeError(
                        f"STL export failed: {export_result.error_message}"
                    )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - step_start) * 1000
            logger.error(
                "Pipeline step %d (%s) failed after %.2fms: %s",
                step_num,
                step_key,
                elapsed_ms,
                e,
                exc_info=True,
            )
            # Freeze progress bar at current position, show error
            status_text.empty()
            st.error(
                f"Generation failed at {step_label.split(':')[0]}: {e}"
            )
            # Store error in session state, preserve previous mesh
            st.session_state["generation_error"] = str(e)
            return

        elapsed_ms = (time.perf_counter() - step_start) * 1000
        logger.debug(
            "Pipeline step %d (%s) exit - elapsed=%.2fms",
            step_num,
            step_key,
            elapsed_ms,
        )

        if elapsed_ms > 2000:
            logger.info(
                "Step %d (%s) exceeded 2s threshold (%.2fms)",
                step_num,
                step_key,
                elapsed_ms,
            )

    # All steps completed successfully
    progress_bar.progress(100)
    status_text.text("Model generation complete!")
    logger.info(
        "run_generation_pipeline() exit - total_elapsed=%.2fms",
        (time.perf_counter() - pipeline_start) * 1000,
    )

    # Store results in session state
    st.session_state["combined_mesh"] = combined_mesh
    st.session_state["export_result"] = export_result
    st.session_state["generation_error"] = None

    st.success("Model generated successfully!")


# --- Generate Model Button ---
generate_disabled = not validation_result.is_valid
if st.button("Generate Model", disabled=generate_disabled, type="primary"):
    with st.spinner("Generating model..."):
        run_generation_pipeline(params)

# --- Display previous error if present ---
if st.session_state.get("generation_error"):
    st.error(f"Last generation error: {st.session_state['generation_error']}")

# --- 3D Viewer ---
if st.session_state.get("combined_mesh") is not None:
    logger.info("Attempting to render 3D model in viewer")
    try:
        render_model(st.session_state["combined_mesh"])
        logger.info("3D model rendered successfully")
    except Exception as e:
        logger.error(
            "Render failed: %s", e, exc_info=True
        )
        render_placeholder("No valid model available for rendering.")
        st.error(f"3D viewer rendering failed: {e}")
else:
    st.info("Generate a model to see the 3D preview.")

# --- Download Button ---
export_result = st.session_state.get("export_result")
if export_result is not None and export_result.success:
    logger.info(
        "Displaying download button - file_size=%d bytes, face_count=%d",
        len(export_result.stl_bytes),
        export_result.face_count,
    )
    st.download_button(
        label="Download STL File",
        data=export_result.stl_bytes,
        file_name="funnel_model.stl",
        mime="application/octet-stream",
    )
elif export_result is not None and not export_result.success:
    logger.error(
        "Export result indicates failure: %s", export_result.error_message
    )
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
