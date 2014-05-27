# Moonshine
Fast, robust optical music recognition using OpenCL

## Example
    python test_structure.py samples/sonata.png

This detects structural information about the score
(staves, measures, and staff systems)
and overlays the result with the image using matplotlib.

Work is in progress to output music using Music21. Currently, this does
not include any timing information within each measure, since we aren't yet
classifying note stems, beams, or dots.

## Features
* Robust staff detection
* Random forest classification

## Goals
* Accurately detecting page layout (staves and measures) on noisy images
* Heuristic for quickly guessing notes so that the output can then be refined
  using online learning
* Glyph classification using semi-supervised learning on unlabeled scanned
  sheet music available from IMSLP

## Requirements
* Python 2.7+
* NumPy
* PyOpenCL
* Pillow, PyPDF, pylibtiff (PDF loading)
* Pylab (displaying structure)
* Music21
