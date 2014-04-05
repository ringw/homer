# moonshine
Fast, robust optical music recognition using OpenCL

## Example
    python -m moonshine samples/sonata.png

There will eventually be an option to output MusicXML or other data extracted
from the score. Currently, this just detects staves and measures and overlays
the result with the image using matplotlib.

## Features
* Robust staff detection

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
* Pylab (drawing)
