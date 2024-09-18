"""!
    Data IO
    =======

    This module contains functions for reading and writing data.

    @author A. Schaer
    @copyright Magnes AG, (C) 2024.
"""

import dataclasses
import enum
import logging
import os
from typing import Iterator

import numpy as np
from numpy import linalg

from . import cfg

logger = logging.getLogger(__name__)


## Proxy options
class ProxyChoice(str, enum.Enum):
    LUMBAR_X: str = "lumbar-x"
    LUMBAR_Y: str = "lumbar-y"
    LUMBAR_Z: str = "lumbar-z"
    LUMBAR_M: str = "lumbar-magnitude"
    LUMBAR_SUM: str = "lumbar-sum"
    THIGH_X: str = "thigh-x"
    THIGH_Y: str = "thigh-y"
    THIGH_Z: str = "thigh-z"
    THIGH_M: str = "thigh-magnitude"
    THIGH_SUM: str = "thigh-sum"
    SHANK_X: str = "shank-x"
    SHANK_Y: str = "shank-y"
    SHANK_Z: str = "shank-z"
    SHANK_M: str = "shank-magnitude"
    SHANK_SUM: str = "shank-sum"


@dataclasses.dataclass
class _3DSignal:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @property
    def norm(self) -> np.ndarray:
        return linalg.norm(self.asarray(), axis=0)

    @property
    def sum(self) -> np.ndarray:
        return np.sum(self.asarray(), axis=0)

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

    def get_proxy(self, choice: ProxyChoice) -> np.ndarray:
        if choice == ProxyChoice.LUMBAR_X:
            return self.lumbar_xl.x
        elif choice == ProxyChoice.LUMBAR_Y:
            return self.lumbar_xl.y
        elif choice == ProxyChoice.LUMBAR_Z:
            return self.lumbar_xl.z
        elif choice == ProxyChoice.LUMBAR_M:
            return self.lumbar_xl.norm
        elif choice == ProxyChoice.LUMBAR_SUM:
            return self.lumbar_xl.sum
        if choice == ProxyChoice.THIGH_X:
            return self.thigh_xl.x
        elif choice == ProxyChoice.THIGH_Y:
            return self.thigh_xl.y
        elif choice == ProxyChoice.THIGH_Z:
            return self.thigh_xl.z
        elif choice == ProxyChoice.THIGH_M:
            return self.thigh_xl.norm
        elif choice == ProxyChoice.THIGH_SUM:
            return self.thigh_xl.sum
        if choice == ProxyChoice.SHANK_X:
            return self.shank_xl.x
        elif choice == ProxyChoice.SHANK_Y:
            return self.shank_xl.y
        elif choice == ProxyChoice.SHANK_Z:
            return self.shank_xl.z
        elif choice == ProxyChoice.SHANK_M:
            return self.shank_xl.norm
        elif choice == ProxyChoice.SHANK_SUM:
            return self.shank_xl.sum
        else:
            raise ValueError(f"Invalid proxy choice {choice}")


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
        sxl.append([cfg.MG2MPS2 * float(n) for n in nibbles[1:4]])
        txl.append([cfg.MG2MPS2 * float(n) for n in nibbles[4:7]])
        lxl.append([cfg.MG2MPS2 * float(n) for n in nibbles[7:10]])
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
