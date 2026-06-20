# ktc_tools

Python library for simulating TEM diffraction from FIB-milled phase gratings.
Developed in the McMorran Lab at the University of Oregon.

## Installation

Clone the repo and install as an editable package:

```bash
git clone https://github.com/McMorranLab/ktc_tools.git
pip install -e ktc_tools/
```

Editable install means changes to the source files are immediately active — no reinstall needed.

## Usage

```python
from ktc_tools import grating_generation as ggen
from ktc_tools import fourier_diffraction as fd
```

## Modules

- **`grating_generation`** — constructs 1D and 2D grating transmission functions from parameterised depth profiles
- **`fourier_diffraction`** — computes single-grating diffraction patterns and efficiency spectra via Fourier optics
- **`corrected_holography`** — holographic reconstruction utilities
- **`streams`** — generates FIB streamfiles for grating milling patterns

See `ExampleScripts/` for usage notebooks.






