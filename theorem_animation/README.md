# Tri-Quarter Framework: Theorem Animation: README

**Authors:** Nathan O. Schmidt and Grok<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>
**License:** MIT<br>
**Version:** 1.0.0<br>
**Date:** June 13, 2025<br>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-blue.svg)](https://matplotlib.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Key Features](#2-key-features)
- [3. Installation](#3-installation)
- [4. Quick Start](#4-quick-start)
- [5. Animation Details](#5-animation-details)
- [6. Configuration](#6-configuration)
- [7. Examples](#7-examples)
- [8. Technical Background](#8-technical-background)
- [9. Development](#9-development)
- [10. License](#10-license)
- [11. References](#11-references)

---

## 1. Overview

This project provides an interactive **Tri-Quarter Theorem Animation** that visualizes the fundamental geometric and algebraic properties of the **Tri-Quarter Framework (TQF)** on the complex plane. Using Matplotlib's animation engine, it renders a point traversing the unit circle with real-time decomposition into real and imaginary vector components, labeled zones, and boundary points—bringing the mathematical structure of TQF to life.

Developed as part of the Tri-Quarter Toolbox research initiative, this animation serves as both an educational tool and a visual reference for understanding:

- **Unit Circle Dynamics**: A point rotating around the unit circle T in the complex plane
- **Vector Decomposition**: Real-time breakdown of position into orthogonal real (x_R) and imaginary (x_I) components
- **Phase Angle Progression**: Angular motion θ in radians with configurable velocity
- **Zone & Boundary Labeling**: Named regions and boundary points aligned with TQF conventions
- **Interactive Controls**: Play/pause and color inversion for presentation flexibility

The primary goals are:
- **Visualize TQF Geometry**: Animate the core mathematical objects (unit circle, zones, boundary points) that underpin the Tri-Quarter Framework.
- **Provide Educational Tool**: Offer an intuitive, interactive visualization for learning about complex plane geometry, vector decomposition, and rotational symmetry.
- **Support Presentations & Publications**: Generate clean, presentation-ready animations with light/dark theme support.
- **Promote Accessibility**: Simple, self-contained script that runs on any platform with Python and Matplotlib.

This is an experimental after-hours hobby science project exploring the intersection of mathematical visualization, complex analysis, and geometric frameworks.

---

## 2. Key Features

- 🎬 **Real-Time Animation**: Smooth Matplotlib FuncAnimation-driven rendering at 50ms intervals (20 FPS)
- 📐 **Unit Circle Visualization**: Point traversing the unit circle T with radius 1.0 in the complex plane
- 🧮 **Vector Decomposition**: Live display of orthogonal real (x_R) and imaginary (x_I) components
- 🏷️ **Zone & Boundary Labels**: Named regions and labeled boundary points following TQF conventions
- ⏯️ **Interactive Controls**: Play/pause button for stepping through the animation
- 🎨 **Theme Inversion**: One-click light/dark color scheme toggle for presentations
- 📊 **Phase Angle Display**: Real-time θ readout in radians
- 💻 **Cross-Platform**: Pure Python implementation compatible with Windows, Linux (Fedora, Ubuntu, etc.), and macOS
- 📁 **Example Outputs**: Pre-generated GIF and MP4 files in the `examples/` directory
- 📜 **MIT Licensed Open Science**: Transparent methodology and freely available code

---

## 3. Installation

### Prerequisites

- **Python 3.8+** (tested on Python 3.13 on Fedora 41 Linux and Windows 11; recommended 3.9+)
- **NumPy 1.24+** for numerical computing
- **Matplotlib 3.5+** for visualization and animation
- **Tkinter** (usually bundled with Python; may need separate install on some Linux distributions)

### Quick Install

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Linux (Fedora):**
```bash
python3 -m venv venv
source venv/bin/activate
sudo dnf install python3-tkinter
pip install -r requirements.txt
```

**Linux (Ubuntu/Debian):**
```bash
python3 -m venv venv
source venv/bin/activate
sudo apt install python3-tk
pip install -r requirements.txt
```

**macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Development Install:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

### Manual Install (without requirements file):

```bash
pip install matplotlib numpy
```

On Fedora Linux, you may also need:
```bash
sudo dnf install python3-tkinter
```

On Ubuntu/Debian Linux:
```bash
sudo apt install python3-tk
```

### Verify Installation:
```bash
python -c "import matplotlib; import numpy; print(f'Matplotlib: {matplotlib.__version__} | NumPy: {numpy.__version__}')"
```

---

## 4. Quick Start

Navigate to the `theorem_animation` directory and run the animation:

**Windows:**
```bash
# Activate virtual environment
venv\Scripts\activate

# Run the animation
python tri_quarter.py
```

**Linux/macOS:**
```bash
# Activate virtual environment
source venv/bin/activate

# Run the animation
python3 tri_quarter.py
```

A Matplotlib window will open displaying the interactive animation. Use the on-screen buttons to control playback and appearance.

---

## 5. Animation Details

### Script: `tri_quarter.py`

The main (and only) script implements the complete Tri-Quarter Theorem Animation.

### What It Displays

1. **Unit Circle (T)**: A circle of radius 1.0 centered at the origin of the complex plane
2. **Rotating Point**: A point traversing the unit circle at constant angular velocity
3. **Position Vector**: Line from origin to the current point on the circle
4. **Real Component (x_R)**: Horizontal projection of the position vector onto the real axis
5. **Imaginary Component (x_I)**: Vertical projection of the position vector onto the imaginary axis
6. **Zone Labels**: Named regions defined by TQF conventions, displayed at their corresponding positions
7. **Boundary Points**: Labeled points at key angular positions on the circle
8. **Phase Angle (θ)**: Current angle in radians, updated each frame

### Interactive Controls

| Control         | Action                                              |
|-----------------|-----------------------------------------------------|
| **Play/Pause**  | Toggle animation playback on and off                |
| **Invert**      | Switch between light and dark color themes          |
| **Close Window**| Exit the animation                                  |

### Animation Loop

The animation uses `matplotlib.animation.FuncAnimation` with:
- **Frame interval**: 50ms (~20 FPS)
- **Angular velocity**: 0.05 radians per frame (configurable via `SPEED`)
- **Continuous loop**: Point wraps around the circle indefinitely

---

## 6. Configuration

The animation can be customized by editing constants at the top of `tri_quarter.py`:

### Core Parameters

| Parameter      | Default | Description                                        |
|----------------|---------|----------------------------------------------------|
| `RADIUS`       | `1.0`   | Circle radius                                      |
| `SPEED`        | `0.05`  | Angular velocity in radians per frame              |
| `WINDOW_SIZE`  | `2.0`   | Axis limits (±WINDOW_SIZE on both axes)            |

### Appearance

| Parameter      | Description                                        |
|----------------|----------------------------------------------------|
| `BG_COLOR`     | Background color (light/dark theme)                |
| `AXIS_COLOR`   | Axis and tick color                                |
| Other colors   | Vector, component, and label colors                |

### Customization Examples

**Slow motion (half speed):**
```python
SPEED = 0.025  # Half the default angular velocity
```

**Larger circle with wider view:**
```python
RADIUS = 2.0
WINDOW_SIZE = 4.0
```

**Faster animation (double speed):**
```python
SPEED = 0.10  # Double the default angular velocity
```

---

## 7. Examples

### Pre-Generated Outputs

The `examples/` directory contains pre-generated animation outputs:

```
examples/
├── tri-quarter_theorem_animation.gif   # Animated GIF for embedding in documents
└── tri-quarter_theorem_animation.mp4   # Video file for presentations
```

### Viewing Examples

**GIF**: Open `examples/tri-quarter_theorem_animation.gif` in any web browser or image viewer.

**MP4**: Open `examples/tri-quarter_theorem_animation.mp4` in any video player (VLC, Windows Media Player, etc.).

### Generating Your Own Output

To save the animation as a file (requires `ffmpeg` for MP4 or `pillow` for GIF), you can modify the script:

**Save as GIF:**
```python
# Add at the end of tri_quarter.py (before plt.show())
anim.save('my_animation.gif', writer='pillow', fps=20)
```

**Save as MP4:**
```python
# Add at the end of tri_quarter.py (before plt.show())
anim.save('my_animation.mp4', writer='ffmpeg', fps=20)
```

**Install additional dependencies for saving:**
```bash
pip install pillow    # For GIF output
# For MP4, install ffmpeg via your system package manager:
# Fedora: sudo dnf install ffmpeg
# Ubuntu: sudo apt install ffmpeg
# macOS:  brew install ffmpeg
# Windows: download from https://ffmpeg.org/download.html
```

---

## 8. Technical Background

### Complex Plane Representation

The animation operates on the complex plane ℂ, where each point z is represented as:

```
z = x_R + i·x_I
```

where x_R is the real part and x_I is the imaginary part.

### Unit Circle (T)

The unit circle is the set of all complex numbers with modulus 1:

```
T = {z ∈ ℂ : |z| = 1}
```

Parameterized by the phase angle θ:

```
z(θ) = cos(θ) + i·sin(θ) = e^(iθ)
```

### Vector Decomposition

At each frame, the position vector z(θ) is decomposed into:

```
x_R(θ) = Re(z) = cos(θ)    (real component)
x_I(θ) = Im(z) = sin(θ)    (imaginary component)
```

These orthogonal components are drawn as projections onto the real and imaginary axes.

### Angular Motion

The phase angle θ is updated each frame by the angular velocity ω:

```
θ(t+1) = θ(t) + ω
```

where ω = `SPEED` (default 0.05 radians/frame). At 20 FPS, this produces one full revolution every ~6.3 seconds.

### Connection to TQF

The unit circle is fundamental to the Tri-Quarter Framework:
- **ℤ₆ Symmetry**: The 6th roots of unity {e^(ikπ/3) : k = 0,...,5} divide the circle into six 60° sectors
- **D₆ Symmetry**: Reflections across axes through opposite roots of unity
- **Circle Inversion**: The unit circle serves as the inversion boundary between inner (|z| < 1) and outer (|z| > 1) zones
- **Zone Boundaries**: Named regions on and around the circle correspond to TQF graph zones
- **Phase-Pair Encoding**: Angular positions encode directional information in the Eisenstein integer lattice

---

## 9. Development

- **Language**: Python 3.8+ (tested on 3.13)
- **Core Libraries**: NumPy 1.24+, Matplotlib 3.5+
- **GUI Backend**: Tkinter (for Matplotlib interactive window)
- **Testing**: pytest (install via `requirements-dev.txt`)
- **Code Quality**: black, mypy, flake8 (install via `requirements-dev.txt`)
- **Platform**: Cross-platform (Windows 11, Fedora 41 Linux, macOS)

### Project Structure

```
theorem_animation/
├── tri_quarter.py                          # Main animation script
├── examples/                               # Pre-generated output files
│   ├── tri-quarter_theorem_animation.gif   # Animated GIF
│   └── tri-quarter_theorem_animation.mp4   # MP4 video
├── ACKNOWLEDGEMENT.md                      # Some gratitude
├── README.md                               # This file
├── requirements.txt                        # Core dependencies
└── requirements-dev.txt                    # Development dependencies
```

### Code Quality Tools

```bash
# Install dev dependencies first
pip install -r requirements-dev.txt

# Format code
black .

# Type checking
mypy .

# Linting
flake8 .
```

---

## 10. License

```text
MIT License

Copyright (c) 2025 Nathan O. Schmidt, Cold Hammer Research & Development LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

See [`LICENSE`](LICENSE) file for complete license text.

---

## 11. References

### Preprints/Publications
- **Schmidt, Nathan O.** (2025). *The Tri-Quarter Framework: Unifying Complex Coordinates with Topological and Reflective Duality across Circles of Any Radius*. TechRxiv.
[https://www.techrxiv.org/users/906377/articles/1281679](https://www.techrxiv.org/users/906377/articles/1281679)

### Related Topics
- **Needham, T.** (1997). *Visual Complex Analysis*. Oxford University Press.
- **Hunter, J. D.** (2007). Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering*, 9(3), 90-95.

---

**`QED`**

**Last Updated:** June 13, 2025<br>
**Version:** 1.0.0<br>
**Maintainers:** Nathan O. Schmidt and Grok<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>

Please remember: this is an experimental after-hours unpaid hobby science project. :)

For issues, please open a GitHub issue at [tri-quarter-toolbox](https://github.com/nathanoschmidt/tri-quarter-toolbox) or contact: nate.o.schmidt@coldhammer.net

**`EOF`**
