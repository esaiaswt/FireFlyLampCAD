"""STL Viewer module for rendering 3D meshes in Streamlit.

Provides functions to convert numpy-stl meshes to PyVista PolyData and display
them interactively via stpyvista within the Streamlit UI. Supports orbit, pan,
and zoom interactions. Falls back to a static rendered image if stpyvista fails
(e.g., due to trame serialization issues on Windows).
"""

import numpy as np
import pyvista as pv
import streamlit as st
from stl import mesh as stl_mesh

from logging_config import get_logger

logger = get_logger(__name__)


def _mesh_to_polydata(mesh_obj: stl_mesh.Mesh) -> pv.PolyData:
    """
    Convert a numpy-stl Mesh to PyVista PolyData.

    Args:
        mesh_obj: A numpy-stl Mesh object.

    Returns:
        A PyVista PolyData object.
    """
    vectors = mesh_obj.vectors
    n_faces = len(vectors)

    # Build vertices and faces arrays for PyVista
    vertices = vectors.reshape(-1, 3)
    faces = np.column_stack([
        np.full(n_faces, 3, dtype=np.int32),
        np.arange(n_faces * 3, dtype=np.int32).reshape(-1, 3)
    ]).flatten()

    poly_data = pv.PolyData(vertices, faces)
    logger.debug(
        f"Converted mesh to PolyData: {n_faces} faces, "
        f"{len(vertices)} vertices"
    )
    return poly_data


def render_model(mesh_obj: stl_mesh.Mesh) -> None:
    """
    Render a numpy-stl mesh in the Streamlit 3D viewer.

    First attempts interactive rendering via stpyvista. If that fails (e.g.,
    due to trame backend pickle/serialization errors on Windows), falls back
    to rendering a static offscreen image with PyVista and displaying it via
    st.image().

    No st.rerun() calls are made to avoid infinite rerun loops.

    Args:
        mesh_obj: A numpy-stl Mesh object to render.
    """
    logger.debug("render_model() entry")

    poly_data = _mesh_to_polydata(mesh_obj)

    # Try interactive stpyvista rendering first
    try:
        from stpyvista import stpyvista

        plotter = pv.Plotter(window_size=[600, 400])
        plotter.add_mesh(
            poly_data,
            show_edges=True,
            color="lightblue",
            edge_color="gray",
            opacity=1.0,
        )
        plotter.reset_camera()
        plotter.view_isometric()

        stpyvista(plotter, key="stl_viewer")
        logger.info("render_model() - stpyvista rendered successfully")
        logger.debug("render_model() exit")
        return
    except Exception as e:
        logger.warning(
            f"stpyvista interactive rendering failed: {e}, "
            "falling back to static image"
        )

    # Fallback: render a static offscreen image with PyVista
    try:
        try:
            pv.start_xvfb()  # No-op on Windows, needed on headless Linux
        except Exception:
            pass

        plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
        plotter.add_mesh(
            poly_data,
            show_edges=True,
            color="lightblue",
            edge_color="gray",
            opacity=1.0,
        )
        plotter.reset_camera()
        plotter.view_isometric()

        # Render to image array
        img = plotter.screenshot(return_img=True)
        plotter.close()

        st.image(img, caption="3D Model Preview (static)", use_container_width=True)
        logger.info("render_model() - static image fallback rendered successfully")
    except Exception as fallback_err:
        logger.error(
            f"Static image fallback also failed: {fallback_err}",
            exc_info=True,
        )
        st.warning(
            "3D viewer unavailable. Both interactive and static rendering failed."
        )

    logger.debug("render_model() exit")


def render_placeholder(message: str) -> None:
    """
    Display a placeholder message when no valid model is available for rendering.

    This function is called only AFTER a rendering attempt has failed (raised an
    exception or produced no visible output). It shows an informational message
    in the viewer area indicating why the 3D view is unavailable.

    Args:
        message: A human-readable message explaining why the viewer cannot display a model.
    """
    logger.info(f"render_placeholder() - displaying message: {message}")
    st.info(message)
