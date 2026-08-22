"""Streamlit multi-page application entry point for 3D STL Generators.

This is the home/landing page for the application. Individual generators
are available as pages in the sidebar navigation.

Pages:
- 1_Funnel_Generator: Parametric funnel with cylinder sleeve STL generator
- 2_Cup_Holder_Generator: Cup holder with ring, stand, and C-leg STL generator
"""

import streamlit as st

from logging_config import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="3D STL Generators",
    page_icon="\U0001f527",
    layout="wide",
)

# --- Home Page Content ---
st.title("3D STL Generators")
st.markdown(
    "Welcome to the 3D STL Generator suite. Use the sidebar to navigate "
    "between available model generators."
)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Funnel Generator")
    st.markdown(
        "Generate a parametric funnel with adjustable diameters, height, "
        "wall thickness, and an integrated cylinder sleeve."
    )
    st.markdown("**Parameters:** bottom/top diameter, funnel height, wall thickness, sleeve height")

with col2:
    st.subheader("Cup Holder Generator")
    st.markdown(
        "Generate a cup holder with a top ring, vertical stand with chamfered "
        "joints, and a C-shaped base leg for stability."
    )
    st.markdown("**Parameters:** ring diameter, ring/stand wall thickness, total height, leg arc, chamfer size")

st.markdown("---")
st.info("Select a generator from the sidebar navigation to get started.")

logger.info("Home page loaded successfully")
