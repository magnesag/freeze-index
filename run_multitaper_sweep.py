#!/usr/bin/env python3.9 -O
"""!
    Evaluation of the FI for Various Multitaper Parameter Options
    =============================================================

    This script evaluates the multitaper FI for various parameter options on the Daphnet data.
    It is run using multiprocess to speed things up by default. To change this behavior,
    set the `WITH_MULTI_PROCESSING = False`.

    @author A. Schaer, H. Maurenbrecher
    @copyright Magnes AG, (C) 2024.
"""
import enum
import itertools
import json
import logging
import os
import warnings

import matplotlib.pyplot as pltlib
import multiprocessing as mp
import numpy as np

from aux import cfg, dataio, compare
from freezing import freezeindex as frz


import time

logging.basicConfig(level=logging.INFO, force=True, format=cfg.LOGGING_FMT)
logger = logging.getLogger(__name__)

WITH_MULTI_PROCESSING = True

PARAM_RANGES = {
    "dt": np.linspace(2, 10, 17),
    "L": np.arange(2, 9),
    "NW": np.linspace(0.5, 10, 20),
    "LFTF": np.linspace(2, 4, 10),
}
PARAM_NAMES_AND_LABELS = {
    "dt": ("window-width", "$T$ [s]"),
    "L": ("number-of-tapers", "$L$ [--]"),
    "NW": ("bandwidth", "$B$ [--]"),
    "LFTF": ("locomotion-freeze-threshold-frequency", "$f$ [Hz]"),
}
PROXY = dataio.ProxyChoice.SHANK_Y
RES_SUBDIR = os.path.join(cfg.RES_DIR, "param-sweep")
RES_FN = os.path.join(RES_SUBDIR, "fis.json")


class SweepParam(str, enum.Enum):
    T: str = "dt"
    L: str = "L"
    NW: str = "NW"
    LFTF: str = "LFTF"


def setup() -> list[str]:
    """!Setup the environment and parse the CLI arguments

    @return List of datafiles
    """
    if __name__ == "__main__":
        logger.info(__doc__)
    warnings.filterwarnings("error")
    for kk, vv in cfg.PLOT_RC.items():
        pltlib.rc(kk, **vv)

    if not os.path.exists(cfg.RES_DIR):
        os.makedirs(cfg.RES_DIR)

    return dataio.get_files_in_dir(cfg.DATA_DIR, cfg.DAPHNET_FILE_EXTENSION)


def eval_fi(
    t: np.ndarray,
    proxy: np.ndarray,
    fs: float,
    multitaper_kwargs: dict[str, float],
    standardize: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    """!Evaluate FIs

    @param t Time array of raw-data
    @param proxy Proxy signal from which to evaluate the FI
    @param fs sampling frequency
    @param multitaper_kwargs passed to `frz.multitaper()`
    @param standardize Whether to standardize the FI values (Default: True)
    """
    recording_time = t[-1] - t[0]
    fi_t, fi = frz.compute_multitaper_fi(proxy, fs, **multitaper_kwargs)
    res = {"t": fi_t.copy() * recording_time + t[0], "fi": fi.copy()}
    if standardize:
        res["fi"] = compare.standardize(res["fi"])

    return res


def single_file_mutlitaper_sweep(
    filepath: str, standardize: bool, sweeping_param: SweepParam
) -> None:
    """!Evaluate FIs on Daphnet sets

    @param fns Data files (filenames with path)
    @param standardize Whether to standardize the FI values
    @param sweeping_param Parameter being swept
    """
    res = None

    logger.info(f"Working on {os.path.basename(filepath)}")
    _id = os.path.basename(filepath).split(".")[0].lower()
    dest_subdir = os.path.join(RES_SUBDIR, _id)
    if not os.path.exists(dest_subdir):
        os.makedirs(dest_subdir)

    data = dataio.load_daphnet_txt(filepath)
    fs = data.get_fs()
    x = data.get_proxy(PROXY)
    flags = data.flag.copy()

    multitaper_kwargs = cfg.MULTITAPER_STANDARD_KWARGS.copy()
    fis = []
    for pval in PARAM_RANGES[sweeping_param]:
        multitaper_kwargs[sweeping_param] = pval
        logger.info(f"Evaluating FI for p = {pval}")
        try:
            fis.append(
                eval_fi(
                    data.t,
                    x,
                    fs,
                    multitaper_kwargs=multitaper_kwargs,
                    standardize=standardize,
                )
            )

        except RuntimeWarning as e:
            logger.error(
                f"Exception {e} raised during evaluation of {filepath} - skipping file"
            )
            continue

    if len(fis) > 0:
        compare.draw_sweep_comparison(
            data.t,
            PARAM_RANGES[sweeping_param],
            fis,
            PARAM_NAMES_AND_LABELS[sweeping_param],
            dest_subdir,
            flags,
            standardized=standardize,
        )

        res = {
            "_id": _id,
            "res": {
                "p": PARAM_RANGES[sweeping_param].tolist(),
                "fi": [fi["fi"].tolist() for fi in fis],
            },
        }

    return res


def compare_fi_for_multitaper_parametric_sweep(
    fps: list[str], standardize: bool, sweeping_param: SweepParam
):
    """!Compare FI for Multitaper Parametric Sweep"""
    res = {}

    if WITH_MULTI_PROCESSING:
        cpu_count = os.cpu_count() - 1
        logger.info(
            f"Running with Multiprocessing. Grabbing {cpu_count} CPUs for the job."
        )
    else:
        cpu_count = 1

    with mp.Pool(cpu_count) as pool:
        sweep_results = pool.starmap(
            single_file_mutlitaper_sweep,
            zip(fps, itertools.repeat(standardize), itertools.repeat(sweeping_param)),
            chunksize=cpu_count,
        )

        for sr in sweep_results:
            if sr is None:
                continue

            res[sr["_id"]] = sr["res"]

    return res


def main() -> None:
    files = setup()
    res = {}
    for sp in SweepParam:
        logger.info(f"Sweeping {sp}")
        res[sp] = compare_fi_for_multitaper_parametric_sweep(
            fps=files, standardize=False, sweeping_param=sp
        )

    with open(RES_FN, "w") as fp:
        json.dump(res, fp, indent=2)


if __name__ == "__main__":
    start = time.time()

    main()

    print(f"{time.time() - start:.1f}s")
