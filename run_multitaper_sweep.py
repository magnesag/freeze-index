#!/usr/bin/env python3.9 -O
"""!
    Evaluation of the FI for Various Multitaper Parameter Options
    =============================================================

    This script evaluates the multitaper FI for various parameter options on the Daphnet data.

    @author A. Schaer
    @copyright Magnes AG, (C) 2024.
"""
import enum
import json
import logging
import os
import warnings

import matplotlib.pyplot as pltlib
import numpy as np

from aux import cfg, dataio, compare
from freezing import freezeindex as frz

logging.basicConfig(level=logging.INFO, force=True, format=cfg.LOGGING_FMT)
logger = logging.getLogger(__name__)


PARAM_RANGES = {
    "dt": np.linspace(2, 10, 17),
    "L": np.arange(2, 9),
    "NW": np.linspace(0.5, 10, 20),
}
PARAM_NAMES_AND_LABELS = {
    "dt": ("window-width", "$T$ [s]"),
    "L": ("number-of-tapers", "$L$ [--]"),
    "NW": ("bandwidth", "$B$ [--]"),
}
PROXY = dataio.ProxyChoice.SHANK_Y
RES_SUBDIR = os.path.join(cfg.RES_DIR, "param-sweep")
RES_FN = os.path.join(RES_SUBDIR, "fis.json")


class SweepParam(str, enum.Enum):
    T: str = "dt"
    L: str = "L"
    NW: str = "NW"


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


def compare_fi_for_multitaper_parametric_sweep(
    fns: list[str], standardize: bool, sweeping_param: SweepParam
) -> None:
    """!Evaluate FIs on Daphnet sets

    @param fns Data files (filenames with path)
    @param standardize Whether to standardize the FI values
    @param sweeping_param Parameter being swept
    """
    res = {}
    for fn in fns:
        logger.info(f"Working on {os.path.basename(fn)}")
        _id = os.path.basename(fn).split(".")[0].lower()
        dest_subdir = os.path.join(RES_SUBDIR, _id)
        if not os.path.exists(dest_subdir):
            os.makedirs(dest_subdir)

        data = dataio.load_daphnet_txt(fn)
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
                    f"Exception {e} raised during evaluation of {fn} - skipping file"
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
            res[_id] = {
                "p": PARAM_RANGES[sweeping_param].tolist(),
                "fi": [fi["fi"].tolist() for fi in fis],
            }

        if cfg.RUN_ONLY_ONE:
            logger.warning("Run only one is enabled, exiting")
            break

    return res


def main() -> None:
    files = setup()
    res = {}
    for sp in SweepParam:
        logger.info(f"Sweeping {sp}")
        res[sp] = compare_fi_for_multitaper_parametric_sweep(
            fns=files, standardize=False, sweeping_param=sp
        )

    with open(RES_FN, "w") as fp:
        json.dump(res, fp, indent=2)


if __name__ == "__main__":
    main()
