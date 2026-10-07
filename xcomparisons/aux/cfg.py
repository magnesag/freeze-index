"""
FI Evaluation Configuration
===========================

Author:
    A. Schaer
Copyright:
    Magnes AG, (C) 2024.
"""

import enum
import os
from typing import Callable

import numpy as np
import matplotlib.pyplot as pltlib


class WellSuitedRecording(enum.StrEnum):
    S02R01 = "S02R01.txt"
    S03R02 = "S03R02.txt"
    S06R01 = "S06R01.txt"


## Paths
FILE_DIR = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))
ROOT_DIR = os.path.normpath(os.path.join(FILE_DIR, "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data", "dataset")
RES_DIR = os.path.join(ROOT_DIR, "res")

if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)

DAPHNET_FILE_EXTENSION = ".txt"

## Loggign options
LOGGING_FMT = "%(asctime)s|%(levelname)s|%(module)s.%(funcName)s() -> %(msg)s"

## Implementation variants' standard arguments
MULTITAPER_STANDARD_KWARGS = {"dt": 5.0, "L": 4, "NW": 2.5, "LFTF": 3, "nmaf": 5}

## Conversion constants
G = 9.81
MS2S = 1e-3
MG2MPS2 = G * 1e-3

## Run options
RUN_ONLY_ONE: WellSuitedRecording | None = WellSuitedRecording.S02R01
USE_TEX = True

## Plotting options
PLOT_RC = {
    "figure": {"figsize": (10, 5)},
    "savefig": {"format": "pdf", "dpi": 300},
    "font": {"family": "serif", "size": 16},
    "text": {"usetex": USE_TEX},
}
SWEEP_CM = pltlib.cm.YlGnBu_r
COMP_CM = pltlib.cm.YlGnBu
SIMILARITY_CM = pltlib.cm.YlGnBu
MT_COLOR = [0.0, 0.0, 0.0, 1.0]
STANDARDIZED_AX_LIM = (-5, 5)
DIRECT_COMPARISON_MAX_PTS = 250


def generate_n_colors_from_cmap(n: int, cmap: Callable) -> list[list[float]]:
    """Generate n colors from a given colormap

    Parameters
    ----------
    n : int
        Number of colors needed
    cmap : Callable
        Colormap to be sampled, usually a member of pltlib.cm

    Returns
    -------
    list[list[float]]
        List of RGB colors sampled from cmap

    Raises
    ------
    ValueError
        When provided n < 1.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1 ({n} was provided)")

    return cmap(np.linspace(0, 1, n))


## Misc
DIRECTIONS = ("x", "y", "z")
SIDES = ("L", "R")
COLORS = {"L": "navy", "R": "deepskyblue"}
