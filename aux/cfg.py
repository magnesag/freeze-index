"""!
    FI Evaluation Configuration
    ===========================

    @author A. Schaer
    @copyright Magnes AG, (C) 2024.
"""

import os

## Paths
FILE_DIR = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))
ROOT_DIR = os.path.normpath(os.path.join(FILE_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RES_DIR = os.path.join(ROOT_DIR, "res")

if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)

## Loggign options
LOGGING_FMT = "%(asctime)s|%(levelname)s|%(module)s.%(funcName)s() -> %(msg)s"

## Implementation variants' standard arguments
MULTITAPER_STANDARD_KWARGS = {"dt": 5.0, "L": 4, "NW": 2.5}

## Conversion constants
G = 9.81
MS2S = 1e-3
MG2MPS2 = G * 1e-3


## Plotting options
PLOT_RC = {
    "figure": {"figsize": (10, 5)},
    "savefig": {"format": "pdf", "dpi": 300},
    "font": {"family": "serif", "size": 16},
    "text": {"usetex": True},
}

## Misc
DIRECTIONS = ("x", "y", "z")
SIDES = ("L", "R")
COLORS = {"L": "navy", "R": "deepskyblue"}