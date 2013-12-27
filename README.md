# moonshine
Optical music recognition (OMR) in Python

## Goals
Improved OMR, built from the ground up to ensure each stage of processing
is robust on real-world samples. The reference implementation is prototyped
using Python and NumPy, and speed should be competitive with existing
software such as Audiveris, but individual stages can be accelerated
using PyOpenCL, etc.

## Requirements
* numpy
* scipy
* numexpr
* Pillow, PyPDF (PDF loading)
* Pylab (drawing)
