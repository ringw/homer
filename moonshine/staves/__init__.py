from . import base, hough, path

from .base import BaseStaves
from .hough import FilteredHoughStaves
from .path import StablePathStaves

# Default implementation
Staves = StablePathStaves
