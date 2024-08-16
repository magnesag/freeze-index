"""!
    Data IO
    =======

    This module contains functions for reading and writing data.

    @author A. Schaer
    @copyright Magnes AG, (C) 2024.
"""

import dataclasses
import logging
import os
from typing import Iterator

import numpy as np
from numpy import linalg

from . import cfg

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _3DSignal:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @property
    def norm(self) -> np.ndarray:
        return linalg.norm(self.asarray(), axis=0)

    def __iter__(self) -> Iterator[np.ndarray]:
        for attr in "xyz":
            yield getattr(self, attr)

    def asarray(self) -> np.ndarray:
        return np.vstack([d for d in self])


@dataclasses.dataclass
class DaphnetRaw:
    t: np.ndarray
    lumbar_xl: _3DSignal
    thigh_xl: _3DSignal
    shank_xl: _3DSignal
    flag: np.ndarray

    def get_fs(self) -> float:
        return 1 / np.mean(np.diff(self.t))


def get_files_in_dir(path: str, extension: str = ".json") -> list[str]:
    """!Get all files in a given directory

    @param path Directory to check
    @param extension File extension
    @return List of all files in path
    """
    res = []
    root, _, files = next(os.walk(path))
    for file in files:
        if file.endswith(extension):
            res.append(os.path.join(root, file))

    return sorted(res)


def load_daphnet_txt(fn: str) -> DaphnetRaw:
    """!Load a Daphnet dataset file

    This function also converts the data into SI units.

    @param fn Filename to be loaded
    @return Data
    """

    with open(fn, "r") as fp:
        lines = fp.readlines()

    t, lxl, txl, sxl, flag = [], [], [], [], []
    for line in lines:
        nibbles = line.strip().split()
        if len(nibbles) != 11:
            logger.warning(
                "Unexpected line in Daphnet file found. Number of entries is not 11. "
                "Going to next line."
            )
            continue

        t.append(cfg.MS2S * float(nibbles[0]))
        lxl.append([cfg.MG2MPS2 * float(n) for n in nibbles[1:4]])
        txl.append([cfg.MG2MPS2 * float(n) for n in nibbles[4:7]])
        sxl.append([cfg.MG2MPS2 * float(n) for n in nibbles[7:10]])
        flag.append(int(nibbles[10]))

    flag = np.array(flag, dtype=int)
    in_exp_idx = flag > 0

    return DaphnetRaw(
        np.array(t)[in_exp_idx],
        _3DSignal(*np.transpose(np.array(lxl)[in_exp_idx, :])),
        _3DSignal(*np.transpose(np.array(txl)[in_exp_idx, :])),
        _3DSignal(*np.transpose(np.array(sxl)[in_exp_idx, :])),
        flag[in_exp_idx],
    )
