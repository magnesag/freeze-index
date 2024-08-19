"""!
    Compare Module
    ==============

    This module implements functions for the comparison of FIs computed using different methds.

    @author A. Schaer
    @copyright Magnes AG, (C) 2024.
"""

import dataclasses
import os

import matplotlib.pyplot as pltlib
import numpy as np
from sklearn import metrics

from aux import cfg


@dataclasses.dataclass
class ComparisonMetrics:
    """!Comparison metrics for FIs"""

    mad: np.ndarray
    rho: np.ndarray
    r2: np.ndarray
    names: list[str]

    def __post_init__(self) -> None:
        """!Post-initialization"""
        self._n = len(self.names)

    def visualize(self, dest: str = None) -> None:
        """!Visualize the comparison metrics"""
        minmad = np.nanmin(self.mad)
        maxmad = np.nanmax(self.mad)
        fig, axs = pltlib.subplots(figsize=(8, 8))
        img = pltlib.imshow(
            self.mad,
            aspect="equal",
            origin="upper",
            vmin=minmad,
            vmax=maxmad,
            cmap="Wistia",
        )
        fig.colorbar(img, ax=axs, label="MAD [-]", shrink=0.73)
        TEXT_FNTSZ = min(cfg.PLOT_RC["font"]["size"], 64 / self._n)
        font_dicts = (
            {"weight": "bold", "color": "white", "size": TEXT_FNTSZ},
            {"size": TEXT_FNTSZ},
        )
        for kk in range(self._n):
            for jj in range(self._n):
                if jj == kk:
                    continue

                if jj > kk:
                    txt = "\n".join(
                        [
                            rf"$\rho$ = {self.rho[kk,jj]:.2f}",
                            rf"$R^2$ = {self.r2[kk,jj]:.2f}",
                        ]
                    )
                else:
                    txt = rf"{self.mad[kk,jj]:.2f}"

                for fd in font_dicts:
                    pltlib.text(kk, jj, txt, fontdict=fd, ha="center", va="center")

        axs.set_xticks([kk for kk in range(self._n)])
        axs.set_xticklabels(
            [case.title() for case in self.names], rotation=90, ha="right"
        )
        axs.set_yticks([kk for kk in range(self._n)])
        axs.set_yticklabels([case.title() for case in self.names])
        fig.tight_layout()
        if dest is None:
            fig.savefig(f"similarity-matrix")
        else:
            fig.savefig(os.path.join(dest, f"similarity-matrix"))


def standardize(x: np.ndarray) -> np.ndarray:
    """!Standardize a vector/time series

    @param x The vector/time series to be standardized.
    @return The standardized vector/time series.
    """
    return (x - np.nanmean(x)) / np.nanstd(x)


def resample_to_n_samples(x: np.ndarray, n: int) -> np.ndarray:
    """!Resample a vector/time series to a specified number of samples

    @param x The input vector/time series to be resampled.
    @param n The desired number of samples in the output.
    @return The resampled vector/time series with n samples.
    """
    original_length = len(x)
    if original_length == n:
        return x

    t = np.linspace(0, 1, n)
    tp = np.linspace(0, 1, original_length)
    return np.interp(t, tp, x)


def compare_signals(xs: np.ndarray, names: list[str]) -> ComparisonMetrics:
    """!Compare signals

    @param xs The signals to be compared: each _row_ is a signal/variable
    @param names The names of the signals.
    @return TBD
    """
    n = len(names)
    rho = np.corrcoef(xs)
    r2 = np.zeros((n, n))
    mad = np.zeros((n, n))
    for ii in range(n):
        for jj in range(ii + 1, n):
            r2[ii, jj] = metrics.r2_score(xs[ii], xs[jj])
            mad[ii, jj] = metrics.mean_absolute_error(xs[ii], xs[jj])

    r2 += r2.T
    mad += mad.T
    np.fill_diagonal(r2, 1.0)

    return ComparisonMetrics(mad, rho, r2, names)


def draw_all_comparisons(
    xs: dict[str, np.ndarray],
    metrics: ComparisonMetrics,
    names: list[str],
    dest: str = None,
):
    """!Plot direct comparisons between different proxies

    @param xs Dictionary of proxy values
    @param metrics Comparison metrics
    @param dest Image file destination (directory)
    """
    n_variants = xs.shape[0]
    fig, axs = pltlib.subplots(
        n_variants, n_variants, figsize=(12, 12), sharex=True, sharey=True
    )
    axs[0, 0].set(xlim=cfg.STANDARDIZED_AX_LIM, ylim=cfg.STANDARDIZED_AX_LIM)
    DOWNSAMPLE = max(int(len(xs[0]) / cfg.DIRECT_COMPARISON_MAX_PTS), 1)
    for ii in range(n_variants):
        for jj in range(n_variants):
            if ii > jj:
                axs[ii, jj].plot(
                    xs[ii][::DOWNSAMPLE],
                    xs[jj][::DOWNSAMPLE],
                    "o",
                    alpha=0.5,
                    c="deepskyblue",
                )
                axs[ii, jj].set(aspect="equal")
                axs[ii, jj].plot(
                    cfg.STANDARDIZED_AX_LIM,
                    cfg.STANDARDIZED_AX_LIM,
                    "--",
                    color="black",
                )
                axs[ii, jj].grid(True)

            elif ii < jj:
                axs[ii, jj].text(
                    0.0,
                    0.0,
                    "\n".join(
                        [
                            rf"$\rho$ = {metrics.rho[ii,jj]:.2f}",
                            f"$R^2$ = {metrics.r2[ii,jj]:.2f}",
                            f"MAD = {metrics.mad[ii,jj]:.2f}",
                        ]
                    ),
                    va="center",
                    ha="center",
                )

            else:
                axs[ii, jj].text(
                    0.0,
                    0.0,
                    f"{names[ii].title()}",
                    ha="center",
                    va="center",
                )
                axs[ii, jj].axis("off")

    for ii in range(n_variants):
        axs[ii, 0].set_ylabel(f"{names[ii].title()} FI")
        axs[-1, ii].set_xlabel(f"{names[ii].title()} FI")

    fig.tight_layout()
    if dest is None:
        fig.savefig("direct-comparisons")
    else:
        fig.savefig(os.path.join(dest, "direct-comparisons"))


def overlay(
    t: np.ndarray,
    estimates: dict[str, np.ndarray],
    flag: np.ndarray,
    dest: str = None,
    standardized: bool = False,
):
    """Overlay estimates of Freeze Index (FI) from different methods on a single plot.

    This function creates a plot that overlays FI estimates from various methods,
    highlighting periods of FOG with gray shading. It saves the resulting plot as an image file.

    Side effects
    - Creates and saves a matplotlib figure as an image file.
    - The filename is either 'fi-overlay-standardized' or 'fi-overlay', depending on
      the 'standardized' parameter.
    - If 'dest' is provided, the file is saved in that directory; otherwise, it's saved
      in the current working directory.

    Notes:
    - The 'multitaper' method, if present in estimates, is plotted with a thicker black line
      and brought to the front of the plot.
    - FOG periods are highlighted with gray shading on the plot.
    - The legend is positioned above the plot for clarity.

    @param t 1D array of time values corresponding to the FI estimates.
    @param estimates A dictionary where keys are method names (e.g., 'multitaper') and values are  dictionaries containing 't' (time) and 'fi' (freeze index) arrays.
    @param flag 1D boolean array indicating the presence of FOG (True) or absence (False) at each time point.
    @param dest Destination directory for saving the plot. If None, saves in the current directory. Default is None.
    @param standardized If True, standardized FI values are assumed and includes this in the filename and y-axis label. Default is False.
    """
    YLABEL = "Standardized FI [-]" if standardized else "FI [-]"
    fn = f"fi-overlay-standardized" if standardized else f"fi-overlay"
    n = len(estimates)
    if "multitaper" in estimates.keys():
        n -= 1

    colors = iter(cfg.generate_n_colors_from_cmap(n))
    fig, axs = pltlib.subplots()
    for case, vals in estimates.items():
        kwargs = {"label": case.title(), "ls": "--"}
        if case == "multitaper":
            kwargs.update({"lw": 2, "c": "black", "zorder": 10, "ls": "--"})
        else:
            kwargs.update({"c": next(colors)})

        axs.plot(vals["t"], vals["fi"], **kwargs)

    fog_starts = np.arange(len(flag) - 1)[np.diff(flag) > 0]
    fog_stops = np.arange(len(flag) - 1)[np.diff(flag) < 0]
    for start, stop in zip(fog_starts, fog_stops):
        pltlib.axvspan(t[start], t[stop], fc="gray", alpha=0.5)

    axs.grid(True)
    axs.set(xlabel="Recording time [s]", xlim=(t[0], t[-1]), ylabel=YLABEL)
    if standardized:
        axs.set_ylim(cfg.STANDARDIZED_AX_LIM)

    axs.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    if dest is None:
        fig.savefig(fn)
    else:
        fig.savefig(os.path.join(dest, fn))


def compare_fis(
    t: np.ndarray,
    estimates: dict[str, dict[str, np.ndarray]],
    dest: str,
    flag: np.ndarray,
    standardized: bool = True,
):
    """!Compare FIs"""

    n = max(len(case["fi"]) for case in estimates.values())
    xs = np.zeros((len(estimates), n))
    names = []
    for ii, (variant, val) in enumerate(estimates.items()):
        xs[ii] = resample_to_n_samples(val["fi"], n)
        names.append(variant)

    comparison_metrics = compare_signals(xs, names)
    comparison_metrics.visualize(dest)
    overlay(t, estimates, flag, dest, standardized)
    draw_all_comparisons(xs, comparison_metrics, names, dest)
    pltlib.close("all")
