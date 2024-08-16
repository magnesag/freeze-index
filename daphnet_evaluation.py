#!/usr/bin/env python3.9 -O
"""!
    Evaluation of the FI Computation on Daphnet Data
    ================================================

    This script evaluates the FI computation on the Daphnet data.

    @author A. Schaer
    @copyright Magnes AG, (C) 2024.
"""
import enum
import logging
import os

import matplotlib.pyplot as pltlib
import numpy as np

from aux import cfg, dataio, compare

from freezing import freezeindex as frz

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
logging.basicConfig(level=logging.INFO, force=True, format=cfg.LOGGING_FMT)
logger = logging.getLogger(__name__)


class ProxyChoice(str, enum.Enum):
    SHANK_X: str = "shank-x"
    SHANK_Y: str = "shank-y"
    SHANK_Z: str = "shank-z"
    SHANK_M: str = "shank-magnitude"


def setup() -> list[str]:
    """!Setup the environment and parse the CLI arguments

    @return Parsed CLI arguments
    """
    logger.info(__doc__)
    for kk, vv in cfg.PLOT_RC.items():
        pltlib.rc(kk, **vv)

    if not os.path.exists(cfg.RES_DIR):
        os.makedirs(cfg.RES_DIR)

    return dataio.get_files_in_dir(cfg.DATA_DIR, ".txt")


def eval_fis(
    t: np.ndarray,
    proxy: np.ndarray,
    fs: float,
    standardize: bool = True,
) -> None:
    """!Evaluate FIs"""
    recording_time = t[-1] - t[0]

    res = {}
    for case in frz.VARIANTS:
        logger.debug(f"Evaluating {case}")
        fi_t, fi = frz.compute_fi_variant(proxy, fs, case)
        res[case] = {"t": fi_t.copy() * recording_time + t[0], "fi": fi.copy()}
        if standardize:
            res[case]["fi"] = compare.standardize(res[case]["fi"])

    return res


def compare_fis(
    t: np.ndarray,
    estimates: dict[str, dict[str, np.ndarray]],
    dest: str,
    flag: np.ndarray,
    standardized: bool = True,
):
    """!Compare FIs"""

    # TODO Reintroduce and refactor
    # YLABEL = "Standardized FI [-]" if standardize else "FI [-]"
    # fn = f"fi-comparison-standardized" if standardize else f"fi-comparison"
    # fig, axs = pltlib.subplots()
    # for case, vals in estimates.items():
    #     kwargs = {"label": case.title()}
    #     if case == "multitaper":
    #         kwargs.update({"lw": 3, "c": "black", "zorder": 10})

    #     axs.plot(vals["t"], vals["fi"], **kwargs)

    # fog_starts = np.arange(len(flag) - 1)[np.diff(flag) > 0]
    # fog_stops = np.arange(len(flag) - 1)[np.diff(flag) < 0]
    # for start, stop in zip(fog_starts, fog_stops):
    #     pltlib.axvspan(t[start], t[stop], fc="gray", alpha=0.5)

    # axs.grid(True)
    # axs.set(xlabel="Recording time [s]", xlim=(t[0], t[-1]), ylabel=YLABEL)
    # axs.legend(loc="lower left", bbox_to_anchor=(0, 1), ncols=len(estimates))
    # fig.tight_layout()
    # fig.savefig(os.path.join(dest, fn))

    n = max(len(case["fi"]) for case in estimates.values())
    xs = np.zeros((len(estimates), n))
    names = []
    for ii, variant in enumerate(frz.VARIANTS):
        xs[ii] = compare.resample_to_n_samples(estimates[variant]["fi"], n)
        names.append(variant)

    comparison = compare.compare_signals(xs, names)
    comparison.visualize(dest)


def main(fns: list[str], standardize, proxy_choice: ProxyChoice) -> None:
    """!Evaluate FIs on Daphnet sets"""

    for fn in fns[:1]:
        logger.info(f"Working on {os.path.basename(fn)}")
        data = dataio.load_daphnet_txt(fn)
        fs = data.get_fs()
        if proxy_choice == ProxyChoice.SHANK_X:
            x = data.shank_xl.x
        elif proxy_choice == ProxyChoice.SHANK_Y:
            x = data.shank_xl.y
        elif proxy_choice == ProxyChoice.SHANK_Z:
            x = data.shank_xl.z
        elif proxy_choice == ProxyChoice.SHANK_M:
            x = data.shank_xl.norm

        _id = os.path.basename(fn).split(".")[0].lower()
        dest_subdir = os.path.join(cfg.RES_DIR, proxy_choice.value, _id)
        if not os.path.exists(dest_subdir):
            os.makedirs(dest_subdir)

        logger.debug(f"Sampling frequency detected to be {fs:.2f} Hz")
        fis = eval_fis(data.t, x, fs, standardize=standardize)
        compare_fis(data.t, fis, dest_subdir, data.flag, standardized=standardize)
        pltlib.close("all")


if __name__ == "__main__":
    files = setup()
    PROXY_CHOICES = [
        ProxyChoice.SHANK_X,
        ProxyChoice.SHANK_Y,
        ProxyChoice.SHANK_Z,
        ProxyChoice.SHANK_M,
    ]
    for choice in PROXY_CHOICES:
        logger.info(f"Evaluating FIs on proxy {choice}")
        for std in (True, False):
            main(fns=files, standardize=std, proxy_choice=choice)
            if cfg.RUN_ONLY_ONE:
                import sys

                logger.warning("Run only one is enabled, exiting")
                sys.exit(0)
