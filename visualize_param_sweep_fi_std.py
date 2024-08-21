#!/usr/bin/env python3.9
"""!
    Visualize the FI's Standard Deviation as a Function of Parameter Values
    =======================================================================

    @author A. Schaer
    @copyright Magnes AG, (C) 2024.
"""

import json
import logging
import os

import matplotlib.pyplot as pltlib
import numpy as np

import run_multitaper_sweep as rms
from aux import cfg

ID_OF_INTEREST = "s01r01"

logger = logging.getLogger(__name__)


def load_data() -> dict[str, list[float]]:
    """!Load the data from the results file"""
    with open(rms.RES_FN, "r") as fp:
        all_res = json.load(fp)

    res = {p: all_res[p][ID_OF_INTEREST]["p"] for p in all_res.keys()}
    aggregated_res = res.copy()
    res.update(
        {
            f"fi-sigma-{p}": all_res[p][ID_OF_INTEREST]["fi-sigma"]
            for p in all_res.keys()
        }
    )
    aggregated_res.update(
        {
            f"fi-sigma-{p}": [all_res[p][_id]["fi-sigma"] for _id in all_res[p].keys()]
            for p in all_res.keys()
        }
    )
    print(aggregated_res)
    return res, aggregated_res


def normalize(x: np.ndarray) -> np.ndarray:
    minx = np.min(x)
    return (x - minx) / (np.max(x) - minx)


def visualize_data(data: dict[str, list[float]]):
    """!Visualize the data"""

    colors = iter(
        cfg.generate_n_colors_from_cmap(len(rms.SweepParam), pltlib.cm.viridis)
    )
    MARKERS = iter("os^")
    fig, axs = pltlib.subplots()
    for sp in rms.SweepParam:
        x = normalize(np.array(data[sp]))
        color = next(colors)
        axs.plot(
            x,
            data[f"fi-sigma-{sp}"],
            label=rms.PARAM_NAMES_AND_LABELS[sp][1],
            c=color,
            marker=next(MARKERS),
            mfc="white",
            mec=color,
            lw=3,
            ms=10,
            mew=3,
        )

    axs.legend(loc="upper left", bbox_to_anchor=(1, 1))
    axs.grid(True)
    axs.set(
        xlim=(0, 1),
        xlabel="Normalized parameter value $p$ [--]",
        ylabel=r"$\sigma($FI$)$ [--]",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(rms.RES_SUBDIR, "fi-sigma-sweep-single"))
    pltlib.show()


def main():
    logger.info(__doc__)
    _ = rms.setup()
    data_of_special_interest, aggregated_res = load_data()
    visualize_data(data_of_special_interest)


if __name__ == "__main__":
    main()
