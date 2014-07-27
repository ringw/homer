# Moonshine
Fast, robust optical music recognition using OpenCL

## Example
    python show_structure.py samples/sonata.png

This detects structural information about the score
(staves, measures, and staff systems)
and overlays the result with the image using matplotlib.

Work is in progress to output music using Music21. Currently, this does
not include any timing information within each measure, since we aren't yet
classifying note stems, beams, or dots.

## Features
* Robust structure (staves, systems, and measures) detection
* Random forest glyph classification

## Goals
* Glyph classification using semi-supervised learning on unlabeled scanned
  sheet music available from IMSLP

## Requirements
* Python 2.7+
* NumPy
* PyOpenCL
* PyFFT
* Pillow, PyPDF, pylibtiff (PDF loading)
* Pylab (displaying structure)
* Music21
