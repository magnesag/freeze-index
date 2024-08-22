#!/usr/bin/env python3.9
"""!
    Visualize FI Computation Variants Comparison Metrics
    ====================================================

    @author A. Schaer
    @copyright Magnes AG, (C) 2024.
"""

import json
import logging
import os
import itertools
from typing import Any

import matplotlib.pyplot as pltlib
import matplotlib.colors as mcolors
import numpy as np
from scipy import stats

from aux import cfg

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
logging.basicConfig(level=logging.INFO, force=True, format=cfg.LOGGING_FMT)
logger = logging.getLogger(__name__)

METRICS = ("mad", "rho", "r2")
SIGNIFICANCE = 0.01


def load_and_sort_comp_res(subdir: str) -> dict:
    """!Load and sort results

    Metrics are sorted into pair-wise comparisons. For each metric and each pair,
    an array is retured.

    @note All keys/names operations are pretty convoluted due to uncertainty in
    dictionary key sequences. The implementation should be robust againts different
    files having led to different sequence of variants.
    """
    res = {metric: None for metric in METRICS}
    pairs = []
    with open(os.path.join(subdir, "comp-res.json"), "r") as fp:
        data = json.load(fp)

    names = None
    for values in data.values():
        if names is None:
            names = values["names"]
            pairs = ["-".join(combo) for combo in itertools.combinations(names, 2)]
            logger.debug(pairs)
            for metric in METRICS:
                res[metric] = {pair: [] for pair in pairs}

        for metric in METRICS:
            for ii in range(len(names)):
                for jj in range(ii + 1, len(names)):
                    pair = f"{names[ii]}-{names[jj]}"
                    if not pair in pairs:
                        pair = f"{names[jj]}-{names[ii]}"

                    res[metric][pair].append(values[metric][ii][jj])

    return res


def load_full_res(subdir: str) -> dict[str, list[float]]:
    with open(os.path.join(subdir, "all-fis.json"), "r") as fp:
        data = json.load(fp)

    return data


def stat_test(pairwise_metric: dict[str, list[float]]) -> dict[str,]:
    """!Perform statistical test on given metric

    Compute the p-value for all pair-combinations. The null-hypothesis is that the pairs have
    the same underlying errors, i.e. that one variant is not significantly different from any
    other variant. The significance level is 0.05.

    The Wilcoxon signed-rank test is used.

    @param pairwise_metric
    @return
    """
    pairs = list(pairwise_metric.keys())
    res = {}
    for p1 in range(len(pairs)):
        for p2 in range(p1 + 1, len(pairs)):
            key = f"{pairs[p1]}-v-{pairs[p2]}"
            res[key] = stats.wilcoxon(
                pairwise_metric[pairs[p1]], pairwise_metric[pairs[p2]]
            )

    return res


def draw_dists(
    pairwise_metric: dict[str, list[float]],
    tstats: dict[str, Any],
    metric: str,
    dest: str,
):
    """!Draw pairwise distance distributions"""
    pairs = list(pairwise_metric.keys())
    FS = 14
    pmask = np.zeros((len(pairs), len(pairs)))
    fig, axs = pltlib.subplots(ncols=len(pairs), nrows=len(pairs), figsize=(2 * FS, FS))
    for p1 in range(len(pairs)):
        for p2 in range(len(pairs)):
            if p1 > p2:
                kde1f = stats.gaussian_kde(pairwise_metric[pairs[p1]])
                kde2f = stats.gaussian_kde(pairwise_metric[pairs[p2]])
                xmin = min(
                    np.min(pairwise_metric[pairs[p1]]),
                    np.min(pairwise_metric[pairs[p2]]),
                )
                xmax = max(
                    np.max(pairwise_metric[pairs[p1]]),
                    np.max(pairwise_metric[pairs[p2]]),
                )
                xr = xmax - xmin
                x = np.linspace(xmin - 0.2 * xr, xmax + 0.2 * xr, 1000)
                kde1 = kde1f(x)
                kde2 = kde2f(x)

                axs[p1, p2].plot(x, kde1, label=pairs[p1].title())
                axs[p1, p2].plot(x, kde2, label=pairs[p2].title())
                testkey = f"{pairs[p2]}-v-{pairs[p1]}"
                pval = tstats[testkey].pvalue
                tval = rf"$p = {pval:.2f}$" if pval >= 0.01 else "$p < 0.01$"
                axs[p1, p2].set_title(tval)
                if pval > SIGNIFICANCE:
                    pmask[p1, p2] = 1.0
                    pmask[p2, p1] = 1.0

            elif p1 < p2:
                testkey = f"{pairs[p1]}-v-{pairs[p2]}"
                axs[p1, p2].set_axis_off()

            else:
                axs[p1, p2].text(0, 0, f"{pairs[p1].title()}", va="center", ha="center")
                axs[p1, p2].set(xlim=(-1, 1), ylim=(-1, 1))
                axs[p1, p2].set_axis_off()

    fig.tight_layout()
    fig.savefig(os.path.join(dest, f"{metric}-kdes"))
    pltlib.close(fig)

    cm = pltlib.cm.gray
    cm_list = [cm(ii) for ii in range(cm.N)]
    cmap = mcolors.LinearSegmentedColormap.from_list("extrema", cm_list, 2)

    fig, axs = pltlib.subplots(figsize=(8, 8))
    img = axs.imshow(pmask, cmap=cmap, aspect="equal", origin="lower")
    cb = fig.colorbar(img, ax=axs, label="Null-hypothesis rejection", shrink=0.68)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["No", "Yes"])
    axs.set_xticks([ii for ii in range(len(pairs))])
    axs.set_xticklabels([p.title() for p in pairs], rotation=45, ha="right", va="top")
    axs.set_yticks([ii for ii in range(len(pairs))])
    axs.set_yticklabels([p.title() for p in pairs])
    fig.tight_layout()
    fig.savefig(os.path.join(dest, f"{metric}-diff-significance"))
    pltlib.close(fig)

    multitaper_entries_idx = [
        ii for ii in range(len(pairs)) if "multitaper" in pairs[ii]
    ]
    logger.debug(f"Pairs: {pairs}, MT entries idx: {multitaper_entries_idx}")
    multitaper_rows = np.concatenate([pmask[ii] for ii in multitaper_entries_idx])
    logger.debug(len(multitaper_rows))
    logger.debug(np.sum([ii for ii in multitaper_entries_idx]))

    logger.debug(f"MT significance entries: {multitaper_rows}")
    multitaper_nh_rejection_rate = (
        0.5
        * np.sum(multitaper_rows)
        / (len(multitaper_rows) - len(multitaper_entries_idx))
    )
    null_hypothesis_rejection_rate = (
        0.5 * np.sum(pmask) / (0.5 * (pmask.size - len(pairs)))
    )
    logger.info(
        f"Null-hypothesis rejeciton rate: {null_hypothesis_rejection_rate*100:.2f}%"
    )
    logger.info(
        f"Multitaper null-hypothesis rejeciton rate: {multitaper_nh_rejection_rate*100:.2f}%"
    )


def visualize_comparison_metrics_res(subdir: str):
    data = load_and_sort_comp_res(subdir)
    logger.debug(data)
    testres = {}
    for metric in METRICS:
        logger.info(f"Performing statistical test on {metric}")
        testres[metric] = stat_test(data[metric])
        logger.debug(testres[metric])
        draw_dists(data[metric], testres[metric], metric, subdir)


def visualize_full_res(subdir: str):
    logger.info("Full FI distributions visualization")
    logger.info("Loading data")
    data = load_full_res(subdir)

    logger.info("Performing distribution visualization via KDE")
    absmin = 0
    absmax = 0
    kdes = {}
    for variant, values in data.items():
        kdes[variant] = stats.gaussian_kde(values)
        absmin = min(absmin, np.min(values))
        absmax = max(absmax, np.max(values))

    x = np.linspace(*cfg.STANDARDIZED_AX_LIM, 500)
    colors = iter(cfg.generate_n_colors_from_cmap(len(kdes) - 1, pltlib.cm.viridis))
    fig, axs = pltlib.subplots()
    for variant, kde in kdes.items():
        if variant == "multitaper":
            c = "black"

        else:
            c = next(colors)

        y = kde(x)
        axs.fill_between(x, 0, y, facecolor=c, alpha=0.2)
        axs.plot(x, y, c=c, lw=2, label=variant.title())

    axs.legend(loc="upper right")
    axs.set(
        xlim=cfg.STANDARDIZED_AX_LIM,
        xlabel="FI [--]",
        ylabel="Occurrence density [--]",
    )
    axs.set_ylim(ymin=0)
    axs.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(subdir, "fis-dists"))
    pltlib.close(fig)

    logger.info(
        "Performing distribution and similarity visualization with box-plot and p-values"
    )
    variants = list(data.keys())
    nv = len(variants)
    pvalues = np.zeros((nv, nv))
    for ii in range(nv):
        for jj in range(ii + 1, nv):
            test_res = stats.ranksums(data[variants[ii]], data[variants[jj]])
            pvalues[ii, jj] = test_res.pvalue

    pvalues += pvalues.T
    logger.info(f"P-values: {pvalues}")

    fig, axs = pltlib.subplots()
    y = absmin
    dy = 1
    pos = [ii for ii in range(nv)]
    pltlib.boxplot(
        [data[variant] for variant in variants],
        positions=pos,
        widths=0.5,
        meanline=True,
        showfliers=False,
        meanprops={"c": "black"},
    )
    for ii, variant in enumerate(variants):
        for jj in range(ii + 1, nv):
            tval = (
                rf"$p = {pvalues[ii,jj]:.2f}$"
                if pvalues[ii, jj] >= 0.01
                else "$p < 0.01$"
            )
            ls = "-" if pvalues[ii, jj] >= SIGNIFICANCE else "--"
            axs.plot([ii, jj], [y, y], c="black", ls=ls, lw=3)
            # axs.text(0.5 * (ii + jj), y, tval, va="bottom", ha="center")
            y -= dy

    axs.plot([], [], c="black", ls="-", lw=3, label=f"$p \geq {SIGNIFICANCE:.2f}$")
    axs.plot([], [], c="black", ls="--", lw=3, label=f"$p < {SIGNIFICANCE:.2f}$")
    axs.legend(loc="lower left", bbox_to_anchor=(0, 1), ncols=2)

    axs.grid(True)
    axs.set_xticks(pos)
    axs.set_xticklabels([v.title() for v in variants])
    axs.set_yticks(np.arange(*cfg.STANDARDIZED_AX_LIM))
    fig.tight_layout()
    fig.savefig(os.path.join(subdir, "fis-dist-comparison"))
    pltlib.close(fig)


def main():
    SUBDIR = os.path.join(FILE_DIR, "res", "shank-y")
    for kk, vv in cfg.PLOT_RC.items():
        pltlib.rc(kk, **vv)

    visualize_comparison_metrics_res(SUBDIR)
    visualize_full_res(SUBDIR)


if __name__ == "__main__":
    logger.info(__doc__)
    main()
