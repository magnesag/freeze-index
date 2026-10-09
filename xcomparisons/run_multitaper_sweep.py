"""
Evaluation of the FI for Various Multitaper Parameter Options
=============================================================

This script evaluates the multitaper FI for various parameter options on
the Daphnet data. It is run using multiprocess to speed things up by
default. To change this behavior, set ``WITH_MULTI_PROCESSING = False``.

Authors:
    - A. Schaer
    - H. Maurenbrecher
Copyright:
    Magnes AG, (C) 2024.
"""

import enum
import itertools
import json
import logging
import os
import time
import warnings

import matplotlib.pyplot as pltlib
import multiprocessing as mp
import numpy as np

from xcomparisons.aux import cfg, dataio, compare
from freezing import freezeindex as frz

logging.basicConfig(level=logging.INFO, force=True, format=cfg.LOGGING_FMT)
logger = logging.getLogger(__name__)

# NOTE This is incompatible with "fancy style" plots. Enabling multi processing falls back to default plot style.
WITH_MULTI_PROCESSING = False

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
    "LFTF": ("locomotion-freeze-threshold-frequency", "$f_t$ [Hz]"),
}
PROXY = dataio.ProxyChoice.SHANK_Y
RES_SUBDIR = os.path.join(cfg.RES_DIR, "param-sweep")
RES_FN = os.path.join(RES_SUBDIR, "fis.json")


class SweepParam(str, enum.Enum):
    """Multitaper FI parameters available for parametric sweeping.

    Attributes
    ----------
    T : str
        Time window duration ``dt``.
    L : str
        Number of tapers ``L``.
    NW : str
        DPSS half-bandwidth parameter ``NW``.
    LFTF : str
        Locomotion-Freeze-Threshold Frequency ``LFTF``.
    """

    T: str = "dt"
    L: str = "L"
    NW: str = "NW"
    LFTF: str = "LFTF"


def setup() -> list[str]:
    """Set up the environment and return the list of data files.

    Returns
    -------
    list[str]
        List of data file paths.
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
    """Evaluate the multitaper FI on the given proxy signal.

    Parameters
    ----------
    t : np.ndarray
        Time array of raw data.
    proxy : np.ndarray
        Proxy signal from which to evaluate the FI.
    fs : float
        Sampling frequency in Hz.
    multitaper_kwargs : dict[str, float]
        Keyword arguments passed to ``frz.compute_multitaper_fi()``.
    standardize : bool, optional
        Whether to standardize the FI values, by default True.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing ``"t"`` (time) and ``"fi"`` (freeze index)
        arrays.
    """
    recording_time = t[-1] - t[0]
    fi_t, fi = frz.compute_multitaper_fi(proxy, fs, **multitaper_kwargs)
    res = {"t": fi_t.copy() * recording_time + t[0], "fi": fi.copy()}
    if standardize:
        res["fi"] = compare.standardize(res["fi"])

    return res


def single_file_mutlitaper_sweep(
    filepath: str, standardize: bool, sweeping_param: SweepParam
) -> dict:
    """Run a multitaper parameter sweep for a single Daphnet file.

    Parameters
    ----------
    filepath : str
        Path to the data file.
    standardize : bool
        Whether to standardize the FI values.
    sweeping_param : SweepParam
        Parameter to sweep over its predefined range.

    Returns
    -------
    dict or None
        Dictionary with ``"_id"`` and ``"res"`` keys if at least one
        window succeeded; ``None`` if all windows failed.
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
) -> dict:
    """Compare the multitaper FI over a parametric sweep across all files.

    Parameters
    ----------
    fps : list[str]
        Data file paths.
    standardize : bool
        Whether to standardize the FI values.
    sweeping_param : SweepParam
        Parameter to sweep.

    Returns
    -------
    dict
        Sweep results keyed by file identifier.
    """
    res = {}

    if WITH_MULTI_PROCESSING:
        cpu_count = os.cpu_count() - 1
        logger.info(
            f"Running with Multiprocessing. Grabbing {cpu_count} CPUs for the job."
        )
        with mp.Pool(cpu_count) as pool:
            sweep_results = pool.starmap(
                single_file_mutlitaper_sweep,
                zip(
                    fps, itertools.repeat(standardize), itertools.repeat(sweeping_param)
                ),
                chunksize=cpu_count,
            )

            for sr in sweep_results:
                if sr is None:
                    continue

                res[sr["_id"]] = sr["res"]
    else:
        for fp in fps:
            sr = single_file_mutlitaper_sweep(fp, standardize, sweeping_param)
            res[sr["_id"]] = sr["res"]

    return res


def main() -> None:
    """Run the multitaper parametric sweep for all parameters."""
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
