"""
    Simulated Signal Inspection
    ===========================

    @author A. Schaer
    @copyright Magnes AG, (C) 2024
"""

import json
import os
import logging

import numpy as np

from freezing import freezeindex as frz

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
RES_DIR = os.path.join(FILE_DIR, "res", "sim")
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


class CFG:
    t1: float = 100.0
    M: int = 10


LOCOMOTOR_FRANGE = (
    frz.FREQUENCY_RANGE.LOCOMOTOR.value[1] - frz.FREQUENCY_RANGE.LOCOMOTOR.value[0]
)
FREEZING_FRANGE = (
    frz.FREQUENCY_RANGE.FREEZING.value[1] - frz.FREQUENCY_RANGE.FREEZING.value[0]
)
FREEZING_TO_LOCOMOTOR_FRANGE_RATIO = FREEZING_FRANGE / LOCOMOTOR_FRANGE
THEORETICAL_FI = {
    "moore": np.log(100 * FREEZING_TO_LOCOMOTOR_FRANGE_RATIO**2),
    "bachlin": FREEZING_TO_LOCOMOTOR_FRANGE_RATIO,
    "zach": np.log(100 * FREEZING_TO_LOCOMOTOR_FRANGE_RATIO**2),
    "cockx": np.log(
        100
        * (
            (
                frz.FREQUENCY_RANGE.FREEZING_COCKX.value[1]
                - frz.FREQUENCY_RANGE.FREEZING_COCKX.value[0]
            )
            / LOCOMOTOR_FRANGE
        )
        ** 2
    ),
    "multitaper": np.log(100 * FREEZING_TO_LOCOMOTOR_FRANGE_RATIO),
}

SAMPLING_FREQUENCIES = (64.0, 100.0, 256.0, 1000.0)


def run_sims(
    m: int,
) -> tuple[list[dict[str, dict[str, list[float]]]], dict[str, list[float]]]:
    """!Run the white-noise simulations

    @param m Number of evaluations to be run for each sampling frequency
    @return Collection of FI standard deviations and RMSE, grouped by sampling frequency and FI implementation
    """
    latest_res = {}
    res = []
    for fs in SAMPLING_FREQUENCIES:
        logger.info(f"Evaluating fs = {fs} Hz")
        _res = {
            "std": {case: [] for case in frz.VARIANTS},
            "rmse": {case: [] for case in frz.VARIANTS},
        }
        n = int(fs * CFG.t1) + 1
        t = np.linspace(0, CFG.t1, n)
        for _ in range(m):
            wn = np.random.randn(len(t))
            for case in frz.VARIANTS:
                fi_t, fi = frz.compute_fi_variant(wn, fs, case, {"nmaf": 1})
                std = np.std(fi)
                rmse = np.sqrt(np.mean(np.square(THEORETICAL_FI[case] - fi)))
                latest_res[case] = {"t": fi_t.copy(), "fi": fi.copy()}
                _res["std"][case].append(std)
                _res["rmse"][case].append(rmse)
                logger.debug(f"{case}::FI STD: {std:.2f}; RMSE: {rmse:.2f}")

        res.append(_res.copy())

    with open(os.path.join(RES_DIR, "metrics.json"), "w") as fp:
        json.dump(res, fp, indent=2)

    return res, latest_res


def report_error_stats(errors: list[dict[str, dict[str, list[float]]]]):
    """!Print report stats"""
    logger.info("FI Definitions Error Metrics Comparison")
    header = f"{'Sampling Frequency [Hz]':>22} {'Estimation Method':>17} {'STD':>11} {'RMSE':>11}"
    hline = "-" * len(header)
    logger.info(header)
    logger.info(hline)
    for ee, fs in zip(errors, SAMPLING_FREQUENCIES):
        for case in frz.VARIANTS:
            msg = f"{fs:>23.2f} "
            msg += f"{case:>17} "
            for metric in ("std", "rmse"):
                mean = np.mean(ee[metric][case])
                std = np.std(ee[metric][case])
                msg += f"{mean:>5.2f}±{std:>5.2f} "

            logger.info(msg)

        logger.info(hline)


def main():
    logger.info(__doc__)
    logger.info(f"Theoretical FI values: {THEORETICAL_FI}")
    np.random.seed(0)
    errors, fis = run_sims(CFG.M)
    report_error_stats(errors)


if __name__ == "__main__":
    main()
