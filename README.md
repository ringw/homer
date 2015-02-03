# MetaOMR
A fast, reliable sheet music informatics system

## Example
    python show_layout.py samples/sonata.png

This detects layout-related information from the score
(staves, measures, and staff systems)
and overlays the result with the image using matplotlib.

## Features
* Robust layout (staves, systems, and measures) detection
* Random forest glyph classification

## Goals
* Identifying multiple scans and editions of the same piece,
  and automatically selecting higher-quality scans
* Creating a high-level OMR accuracy benchmark

## Requirements
* Python 2.7+
* NumPy
* Reikna
* PyOpenCL or PyCUDA
* PyFFT
* Pillow, PyPDF, pylibtiff (PDF loading)
* Pylab (for displaying data only)
* SciPy (for Kanungo noise parameter estimation)
* Music21 (for classified note export)
