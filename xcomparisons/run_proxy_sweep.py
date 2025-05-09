#!/usr/bin/env python3.9 -O
"""!
Evaluation of the FI for Various Proxy Signals on Daphnet Data
==============================================================

This script evaluates the multitaper FI for various proxy choices on the Daphnet data.

@author A. Schaer
@copyright Magnes AG, (C) 2024.
"""
import json
import logging
import os
import warnings

import matplotlib.pyplot as pltlib
import numpy as np

from xcomparisons.aux import cfg, dataio, compare
from freezing import freezeindex as frz

logging.basicConfig(level=logging.INFO, force=True, format=cfg.LOGGING_FMT)
logger = logging.getLogger(__name__)


def setup() -> list[str]:
    """!Setup the environment and parse the CLI arguments

    @return List of datafiles
    """
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
    standardize: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    """!Evaluate FIs

    @param t Time array of raw-data
    @param proxy Proxy signal from which to evaluate the FI
    @param fs sampling frequency
    @param standardize Whether to standardize the FI values (Default: False)
    """
    recording_time = t[-1] - t[0]
    fi_t, fi = frz.compute_multitaper_fi(proxy, fs)
    res = {"t": fi_t.copy() * recording_time + t[0], "fi": fi.copy()}
    if standardize:
        res["fi"] = compare.standardize(res["fi"])

    return res


def compare_fi_for_proxys(
    fns: list[str], standardize: bool, force: bool = False
) -> None:
    """!Evaluate FIs on Daphnet sets

    @param fns Data files (filenames with path)
    @param standardize Whether to standardize the FI values
    @param force Whether to force computation or try to load from cache (Default: False)
    """

    for fn in fns:
        logger.info(f"Working on {os.path.basename(fn)}")
        data = dataio.load_daphnet_txt(fn)
        fs = data.get_fs()
        fis = {}
        _id = os.path.basename(fn).split(".")[0].lower()
        dest_subdir = os.path.join(cfg.RES_DIR, "proxy-sweep", _id)
        if not os.path.exists(dest_subdir):
            os.makedirs(dest_subdir)
        res_fn = os.path.join(dest_subdir, "comparison.json")

        if force or not os.path.exists(res_fn):
            for pc in dataio.ProxyChoice:
                logger.info(f"Evaluating FI with proxy: {pc}")
                x = data.get_proxy(pc)
                logger.debug(f"Sampling frequency detected to be {fs:.2f} Hz")
                try:
                    fis[pc] = eval_fi(data.t, x, fs, standardize=standardize)
                except RuntimeWarning as e:
                    logger.error(
                        f"Exception {e} raised during evaluation of {fn} - skipping file"
                    )
                    continue

            try:
                comp, _ = compare.compare_fis(
                    data.t, fis, dest_subdir, data.flag, standardized=standardize
                )
                ious = compare.compute_and_visualize_ious(comp, dest_subdir)
                res = {
                    "names": comp.names,
                    "mad": comp.mad.tolist(),
                    "rho": comp.rho.tolist(),
                    "r2": comp.r2.tolist(),
                    "iou": ious,
                }
                with open(res_fn, "w") as fp:
                    json.dump(res, fp, indent=2)

            except ValueError as e:
                logger.error(f"ValueError {e} for {fn} - skipping file")
                continue

        else:
            logger.info("Loading results from cache")
            with open(res_fn, "r") as fp:
                res = json.load(fp)

            comp = compare.ComparisonMetrics(
                np.array(res["mad"]),
                np.array(res["rho"]),
                np.array(res["r2"]),
                res["names"],
            )
            comp.visualize(dest=dest_subdir)
            ious = compare.compute_and_visualize_ious(comp, dest_subdir)

        if cfg.RUN_ONLY_ONE:
            import sys

            logger.warning("Run only one is enabled, exiting")
            sys.exit(0)


def main() -> None:
    files = setup()
    compare_fi_for_proxys(
        fns=files,
        standardize=True,
        force=True,
    )


if __name__ == "__main__":
    main()
