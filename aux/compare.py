"""!
    Compare Module
    ==============

    This module implements functions for the comparison of FIs computed using different methds.

    @author A. Schaer
    @copyright Magnes AG, (C) 2024.
"""

import dataclasses
import itertools
import logging
import os

import matplotlib.pyplot as pltlib
import matplotlib.colors as mcols
import numpy as np
from sklearn import metrics
from scipy import signal

from aux import cfg


logger = logging.getLogger(__name__)


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

    def __iter__(self):
        for x in (self.mad, self.rho, self.r2):
            yield x

    def compute_metrics_iou(self, name: str) -> tuple[float, float, float]:
        """!Compute the IOU of the spanned ranges when leaving the selected case out

        @param name Name of the case to be compared to all others
        @return IOU(MAD), IOU(rho), IOU(R2)
        """
        if name not in self.names:
            raise ValueError(f"Provided case {name} is not in {self.names}")

        res = []
        case_idx = self.names.index(name)
        for values in self:
            case_vals = []
            other_vals = []
            for ii in range(values.shape[0]):
                for jj in range(ii + 1, values.shape[1]):
                    if ii == case_idx or jj == case_idx:
                        case_vals.append(values[ii, jj])
                    else:
                        other_vals.append(values[ii, jj])

            res.append(self.compute_iou(case_vals, other_vals))

        return tuple(res)

    def visualize(self, dest: str = None) -> None:
        """!Visualize the comparison metrics"""
        minmad = np.nanmin(self.mad)
        maxmad = np.nanmax(self.mad)
        figside = 5 / 5 * self._n
        fig, axs = pltlib.subplots(figsize=(figside, figside))
        img = pltlib.imshow(
            self.mad,
            aspect="equal",
            origin="upper",
            vmin=minmad,
            vmax=maxmad,
            cmap="Wistia",
        )
        fig.colorbar(img, ax=axs, label="MAD [-]", shrink=0.73)
        if "multitaper" in self.names:
            TEXT_FNTSZ = min(cfg.PLOT_RC["font"]["size"], 32 / self._n)
        else:
            TEXT_FNTSZ = cfg.PLOT_RC["font"]["size"] * 0.6
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

    @staticmethod
    def compute_iou(seta: list[float], setb: list[float]) -> float:
        """!Compute the IOU of the ranges spanned by set A and set B

        @param seta Elements of A
        @param setb Elements of B
        @return IOU or ranges spanned by A and B
        """
        maxa = max(seta)
        maxb = max(setb)
        mina = min(seta)
        minb = min(setb)
        maxmax = max(maxa, maxb)
        minmin = min(mina, minb)
        minmax = min(maxa, maxb)
        maxmin = max(mina, minb)
        union = maxmax - minmin
        intersection = max(minmax - maxmin, 0)
        return intersection / union if union > 0 else 0.0


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
    @param names Case names
    @param dest Image file destination (directory)
    """
    logger.info("Drawing direct comparison")
    n_variants = xs.shape[0]
    figside = 12 / 5 * n_variants
    fig, axs = pltlib.subplots(
        n_variants, n_variants, figsize=(figside, figside), sharex=True, sharey=True
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


def mark_fog_regions_on_axs(t: np.ndarray, flag: np.ndarray, axs: pltlib.axes):
    """!Mark FOG region on axes

    @param t Time array
    @param flag FOG flag array
    @param axs Axes on whicht to draw FOG regions
    """
    fog_starts = np.arange(len(flag) - 1)[np.diff(flag) > 0]
    fog_stops = np.arange(len(flag) - 1)[np.diff(flag) < 0]
    for start, stop in zip(fog_starts, fog_stops):
        axs.axvspan(t[start], t[stop], fc="gray", alpha=0.5)


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

    @param t 1D array of time values corresponding to the FI estimates.
    @param estimates A dictionary where keys are method names (e.g., 'multitaper') and values are  dictionaries containing 't' (time) and 'fi' (freeze index) arrays.
    @param flag 1D boolean array indicating the presence of FOG (True) or absence (False) at each time point.
    @param dest Destination directory for saving the plot. If None, saves in the current directory. Default is None.
    @param standardized If True, standardized FI values are assumed and includes this in the filename and y-axis label. Default is False.
    """
    logger.info("Drawing ovelray comparison")
    YLABEL = "Standardized FI [-]" if standardized else "FI [-]"
    fn = f"fi-overlay-standardized" if standardized else f"fi-overlay"
    n = len(estimates)
    if "multitaper" in estimates.keys():
        n -= 1

    colors = iter(cfg.generate_n_colors_from_cmap(n, cfg.COMP_CM))
    fig, axs = pltlib.subplots()
    mark_fog_regions_on_axs(t, flag, axs)
    for case, vals in estimates.items():
        kwargs = {"label": case.title(), "ls": "-", "lw": 3, "zorder": 5}
        if case == "multitaper":
            kwargs.update({"lw": 2, "c": "black", "zorder": 10, "ls": "-"})
        elif case == "zach":
            kwargs.update({"c": next(colors), "zorder": 1})
        else:
            kwargs.update({"c": next(colors)})

        axs.plot(vals["t"], vals["fi"], **kwargs)

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


def draw_fi_spectra(estimates: dict[str, np.ndarray], dest: str):
    """!Draw spectra of the FI estimates

    @param estimates FI estimates
    @param dest Destination where to store plot
    """
    logger.info("Drawing spectra")
    spectra = []
    freqs = []
    names = []
    nperseg = 256
    for name, x in estimates.items():
        try:
            fs = 1 / np.mean(np.diff(x["t"]))
            f, X = signal.welch(x["fi"], fs=fs, nperseg=nperseg)
        except UserWarning:
            f, X = [], []

        spectra.append(X.copy())
        freqs.append(f.copy())
        names.append(name.title())

    n = len(estimates)
    if "Multitaper" in names:
        n -= 1

    colors = iter(cfg.generate_n_colors_from_cmap(n, cfg.COMP_CM))
    fig, axs = pltlib.subplots()
    for f, x, name in zip(freqs, spectra, names):
        kwargs = {"label": name.title(), "ls": "-", "lw": 3}
        if name == "Multitaper":
            kwargs.update({"lw": 3, "c": "black", "zorder": 10, "ls": "--"})
        else:
            kwargs.update({"c": next(colors)})

        axs.plot(f, x, **kwargs)

    axs.set(
        xscale="log",
        yscale="log",
        xlabel="Frequency [Hz]",
        ylabel=r"$\rm PSD(FI)$ [--]",
        xlim=(1e-2, 1e1),
        ylim=(1e-5, 1e2),
    )
    axs.grid(True, which="both")
    axs.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fn = "fi-spectra"
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
) -> tuple[ComparisonMetrics, list[str]]:
    """!Compare FIs

    @param t Time array
    @param estimates FI estimates
    @param dest Destination where to store images (plots)
    @param flag FOG signal array
    @param standardized Whether FIs are standardized
    """
    logger.info("Comparing FIs")
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
    draw_fi_spectra(estimates, dest)
    pltlib.close("all")
    return comparison_metrics, names


def draw_sweep_comparison(
    t: np.ndarray,
    param_values: np.ndarray,
    estimates: list[dict[str, np.ndarray]],
    param_name_label: tuple[str, str],
    dest: str = None,
    flags: np.ndarray = None,
    standardized: bool = True,
):
    """!Draw sweep comparison

    @param t Time array
    @param param_values Sweep parameter values
    @param estimates FI estimates for each parameter value
    @param param_name_label Parameter-name and axis-label pairs
    @param dest Destination where to store plots
    @param flags FOG regions signal
    @param standardized Whether the FIs are standardized
    """
    logger.info("Drawing sweep plot")
    fn = f"{param_name_label[0]}-sweep"
    colors = cfg.generate_n_colors_from_cmap(len(param_values), cfg.SWEEP_CM)
    fig, axs = pltlib.subplots()
    for ii, est in enumerate(estimates):
        axs.plot(est["t"], est["fi"], c=colors[ii])

    fig.colorbar(
        pltlib.cm.ScalarMappable(
            norm=mcols.Normalize(param_values[0], param_values[-1]), cmap=cfg.SWEEP_CM
        ),
        ax=axs,
        label=param_name_label[1],
    )
    if not flags is None:
        mark_fog_regions_on_axs(t, flags, axs)
    axs.grid(True)
    axs.set(
        xlim=(estimates[0]["t"][0], estimates[0]["t"][-1]),
        xlabel="Recording time [s]",
        ylabel="FI [--]",
    )

    if standardized:
        axs.set_ylim(cfg.STANDARDIZED_AX_LIM)
        fn += "-standardized"

    fig.tight_layout()
    if not dest is None:
        fn = os.path.join(dest, fn)

    fig.savefig(fn)
    pltlib.close(fig)


def compute_and_visualize_ious(
    comparison: ComparisonMetrics, dest: str
) -> dict[str, list[float]]:
    """!Compute and visualize IOUs"""
    ious = {"mad": [], "rho": [], "r2": []}
    for name in comparison.names:
        mad, rho, r2 = comparison.compute_metrics_iou(name)
        ious["mad"].append(mad)
        ious["rho"].append(rho)
        ious["r2"].append(r2)

    if "multitaper" in [name.value for name in comparison.names]:
        colors = np.vstack(
            [
                cfg.generate_n_colors_from_cmap(len(comparison.names) - 1, cfg.COMP_CM),
                np.array([0, 0, 0, 1]),
            ]
        )
    else:
        colors = np.vstack(
            [
                cfg.generate_n_colors_from_cmap(len(comparison.names) // 3, cfg.COMP_CM)
                for _ in range(3)
            ]
        )

    markers = {"lumbar": "s", "thigh": "^", "shank": "o"}
    labels = {"mad": "IOU(MAD)", "rho": r"IOU($\rho$)", "r2": r"IOU($R^2$)"}

    for combo in itertools.combinations(ious.keys(), 2):
        a, b = combo
        fig, axs = pltlib.subplots(figsize=(10, 8))
        for ii, name in enumerate(comparison.names):
            mk = name.split("-")[0]
            marker = markers[mk] if mk in markers.keys() else "o"
            axs.plot(
                ious[a][ii],
                ious[b][ii],
                marker=marker,
                c=colors[ii],
                ms=20,
                mec="black",
                label=name.title(),
                ls="",
            )

        axs.grid(True)
        axs.set(xlim=(0, 1), ylim=(0, 1), xlabel=labels[a], ylabel=labels[b])
        axs.legend(loc="upper left", bbox_to_anchor=(1, 1))
        fig.tight_layout()
        pltlib.savefig(os.path.join(dest, f"iou-{a}-{b}"))
        pltlib.close(fig)

    return ious
