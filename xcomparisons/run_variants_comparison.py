#!/usr/bin/env python3.9 -O
"""!
Evaluation of the FI Variants on Daphnet Data
=============================================

This script evaluates the variants of the FI definition on the Daphnet data.

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


def eval_fis(
    t: np.ndarray,
    proxy: np.ndarray,
    fs: float,
    standardize: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    """!Evaluate FIs

    @param t Time array of raw-data
    @param proxy Proxy signal from which to evaluate the FI
    @param fs sampling frequency
    @param standardize Whether to standardize the FI values (Default: True)
    @return result in dict fmt
    """
    recording_time = t[-1] - t[0]

    res = {}
    for case in frz.VARIANTS:
        logger.debug(f"Evaluating {case}")
        fi_t, fi = frz.compute_fi_variant(proxy, fs, case)
        res[case] = {"t": fi_t.copy() * recording_time + t[0], "fi": fi.copy()}
        if standardize:
            res[case]["fi"] = compare.standardize(res[case]["fi"])

    return res


def compare_implementations_for_proxy(
    fns: list[str], standardize: bool, proxy_choice: dataio.ProxyChoice
) -> None:
    """!Evaluate FIs on Daphnet sets

    @param fns Data files (filenames with path)
    @param standardize Whether to standardize the FI values
    @param proxy_choice Proxy signal of choice
    """
    cres = {}
    all_fis = {}
    for fn in fns:
        try:
            logger.info(f"Working on {os.path.basename(fn)}")
            data = dataio.load_daphnet_txt(fn)
            fs = data.get_fs()
            x = data.get_proxy(proxy_choice)

            _id = os.path.basename(fn).split(".")[0].lower()
            dest_subdir = os.path.join(cfg.RES_DIR, proxy_choice.value, _id)
            if not os.path.exists(dest_subdir):
                os.makedirs(dest_subdir)

            logger.debug(f"Sampling frequency detected to be {fs:.2f} Hz")
            try:
                fis = eval_fis(data.t, x, fs, standardize=standardize)
            except RuntimeWarning as e:
                logger.error(
                    f"Exception {e} raised during evaluation of {fn} - skipping file"
                )
                continue

            for variant, value in fis.items():
                if variant not in all_fis.keys():
                    all_fis[variant] = value["fi"].copy().tolist()
                else:
                    all_fis[variant] += value["fi"].copy().tolist()

            fres = compare.compare_fis(
                data.t, fis, dest_subdir, data.flag, standardized=standardize
            )
            _ = compare.compute_and_visualize_ious(fres[0], dest_subdir)
            cres[_id] = {
                "names": fres[1],
                "mad": fres[0].mad.tolist(),
                "rho": fres[0].rho.tolist(),
                "r2": fres[0].r2.tolist(),
                "iou": fres[0].compute_metrics_iou("multitaper"),
            }
            with open(os.path.join(dest_subdir, "comp-res.json"), "w") as fp:
                json.dump(cres[_id], fp, indent=2)

            if cfg.RUN_ONLY_ONE:
                logger.warning("Run only one is enabled, exiting")
                break

        except IndexError as e:
            pass

    with open(os.path.join(dest_subdir, "..", "comp-res.json"), "w") as fp:
        json.dump(cres, fp, indent=2)

    with open(os.path.join(dest_subdir, "..", "all-fis.json"), "w") as fp:
        json.dump(all_fis, fp, indent=2)


def main() -> None:
    files = setup()
    for choice in dataio.ProxyChoice:
        logger.info(f"Evaluating FIs on proxy {choice}")
        compare_implementations_for_proxy(
            fns=files, standardize=True, proxy_choice=choice
        )


if __name__ == "__main__":
    main()
