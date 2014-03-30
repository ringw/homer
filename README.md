# moonshine
Fast, robust optical music recognition using OpenCL

## Features
* Robust staff detection

## Goals
* Accurately detecting page layout (staves and measures) on noisy images
* Heuristic for quickly guessing notes so that the output can then be refined
  using online learning
* Glyph classification using semi-supervised learning on unlabeled scanned
  sheet music available from IMSLP

## Requirements
* NumPy
* PyOpenCL
* Pillow, PyPDF, pylibtiff (PDF loading)
* Pylab (drawing)
