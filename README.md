# 3D STL Generators

A Streamlit-based multi-page application for generating parametric 3D models and exporting them as STL files for 3D printing. Includes interactive 3D preview and one-click download.

## Features

- **Multi-Page App** — Navigate between multiple STL generators from the sidebar
- **Parametric Inputs** — Adjust dimensions through intuitive sidebar controls
- **Real-Time Validation** — Inline error messages for invalid geometric constraints
- **3D Interactive Viewer** — Orbit, pan, and zoom the generated model in the browser (powered by PyVista + stpyvista)
- **STL Export & Download** — Export models as binary STL files with one-click download
- **Mesh Validation** — Automatic watertight/manifold checks, degenerate face detection, and normal consistency verification
- **Progress Feedback** — Step-by-step progress bar during model generation
- **Graceful Shutdown** — Built-in button to terminate the application cleanly

## Generators

### Funnel Generator

Design a hollow truncated cone (funnel) with an integrated cylinder sleeve collar. Parametric controls for top/bottom diameters, height, wall thickness, and sleeve dimensions.

### Cup Holder Generator

Design a cup holder consisting of:
- **Top Ring** — Circular ring to hold the cup (inner diameter, wall thickness, height)
- **Vertical Stand** — Connects the ring to the base leg with configurable wall thickness
- **C-Shaped Base Leg** — Partial arc at the bottom for stability (adjustable arc span)
- **Chamfered Joints** — Reinforcing chamfers at ring-to-stand and stand-to-leg joints

## Architecture

The application is organized into three layers:

| Layer | Responsibility | Modules |
|-------|---------------|---------|
| **UI Layer** | Streamlit pages, progress display, 3D viewer, download/shutdown | `app.py`, `pages/`, `stl_viewer.py` |
| **Geometry Engine** | Parametric mesh generation, assembly, validation | `funnel_generator.py`, `cup_holder_generator.py`, `mesh_validator.py`, `parameter_validator.py` |
| **I/O Layer** | Binary STL export and round-trip verification | `stl_exporter.py` |

Supporting infrastructure:
- `logging_config.py` — Rotating file logger (5MB max, 3 backups)
- `tests/` — Unit and property-based tests

## Prerequisites

- **Anaconda** (or Miniconda) with the `firefly` environment
- **Python 3.9+**

## Setup

1. Activate the Anaconda environment:

   ```bash
   conda activate firefly
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

```bash
streamlit run app.py
```

The app will open in your default browser. Use the sidebar navigation to select a generator, configure parameters, and generate/download STL models.

## Parametric Inputs

### Funnel Generator

| Parameter | Default | Constraints |
|-----------|---------|-------------|
| Bottom Diameter | 52.0 mm | > 0, <= 500 mm |
| Top Diameter | 70.0 mm | > 0, <= 500 mm |
| Funnel Height | 30.0 mm | > 0, <= 500 mm |
| Wall Thickness | 1.2 mm | > 0, < min(bottom_diameter, top_diameter) / 2 |
| Sleeve Height | 50.0 mm | > 0, <= 500 mm |
| Sleeve Wall Thickness | 1.2 mm | > 0, < top_diameter / 2 |
| Circumferential Segments | 64 | >= 3 |

### Cup Holder Generator

| Parameter | Default | Constraints |
|-----------|---------|-------------|
| Ring Inner Diameter | 62.0 mm | 10 - 200 mm |
| Ring Wall Thickness | 1.2 mm | 0.4 - 10 mm, < inner diameter / 2 |
| Ring Height | 15.0 mm | 5 - 100 mm |
| Total Height (ring top to leg bottom) | 40.0 mm | 20 - 200 mm, > ring height |
| Stand Wall Thickness | 3.0 mm | 1 - 20 mm |
| Leg Arc Span | 180 degrees | 90 - 330 degrees |
| Chamfer Size | 3.0 mm | 0.5 - 15 mm, < stand height / 2 |
| Circle Segments | 64 | 16 - 128 |

## Testing

Run the full test suite:

```bash
python -m pytest tests/ -v
```

Run property-based tests only:

```bash
python -m pytest tests/test_properties.py -v
```

## File Structure

```
Firefly_Lamp/
├── app.py                       # Streamlit entry point (home page)
├── pages/
│   ├── 1_Funnel_Generator.py    # Funnel STL generator page
│   └── 2_Cup_Holder_Generator.py # Cup holder STL generator page
├── funnel_generator.py          # Funnel + sleeve mesh generation, assembly
├── cup_holder_generator.py      # Cup holder mesh generation (ring, stand, leg, chamfers)
├── mesh_validator.py            # Mesh quality validation (trimesh)
├── parameter_validator.py       # Input validation and data models
├── stl_exporter.py              # Binary STL export and round-trip check
├── stl_viewer.py                # 3D viewer (stpyvista wrapper)
├── logging_config.py            # Rotating log file configuration
├── requirements.txt             # pip dependencies
├── .env                         # Environment secrets (not committed)
├── .gitignore                   # Git ignore rules
├── logs/                        # Rotating log files
│   └── app.log
└── tests/
    ├── test_funnel_generator.py
    ├── test_mesh_validator.py
    ├── test_stl_exporter.py
    ├── test_parameter_validator.py
    └── test_properties.py       # Property-based tests (Hypothesis)
```

## Development

This STL application was developed using [AWS Kiro](https://kiro.dev), an AI-powered IDE built on VS Code.

## License

*License information to be added.*
