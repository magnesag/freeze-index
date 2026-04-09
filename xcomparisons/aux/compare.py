"""
Compare Module
==============

This module implements functions for the comparison of FIs computed using
different methods.

Author:
    A. Schaer
Copyright:
    Magnes AG, (C) 2024.
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

from xcomparisons.aux import cfg


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ComparisonMetrics:
    """Pairwise comparison metrics for a set of FI estimates.

    Attributes
    ----------
    mad : np.ndarray
        Pairwise mean absolute deviation matrix.
    rho : np.ndarray
        Pairwise Pearson correlation coefficient matrix.
    r2 : np.ndarray
        Pairwise coefficient of determination matrix.
    names : list[str]
        Names of the compared FI variants.
    """

    mad: np.ndarray
    rho: np.ndarray
    r2: np.ndarray
    names: list[str]

    def __post_init__(self) -> None:
        """Store the number of compared variants."""
        self._n = len(self.names)

    def __iter__(self):
        """Iterate over the metric matrices (mad, rho, r2)."""
        for x in (self.mad, self.rho, self.r2):
            yield x

    def compute_metrics_iou(self, name: str) -> tuple[float, float, float]:
        """Compute the IOU of the spanned ranges leaving one case out.

        Parameters
        ----------
        name : str
            Name of the case to compare against all others.

        Returns
        -------
        tuple[float, float, float]
            IOU of the MAD, rho, and R² value ranges.

        Raises
        ------
        ValueError
            If ``name`` is not present in ``self.names``.
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
        """Visualize the comparison metrics as a similarity matrix.

        Parameters
        ----------
        dest : str, optional
            Directory where the plot is saved. If ``None``, saves in the
            current working directory, by default None.
        """
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
            cmap=cfg.SIMILARITY_CM,
        )
        fig.colorbar(img, ax=axs, label="MAD [-]", shrink=0.73)
        if "multitaper" in self.names:
            TEXT_FNTSZ = min(cfg.PLOT_RC["font"]["size"], 48 / self._n)
        else:
            TEXT_FNTSZ = cfg.PLOT_RC["font"]["size"] * 0.6
        font_dicts = (
            {"weight": "heavy", "color": "black", "size": TEXT_FNTSZ, "rotation": 45},
            {"size": TEXT_FNTSZ, "color": "white", "weight": "light", "rotation": 45},
        )
        for kk in range(self._n):
            for jj in range(self._n):
                if jj == kk:
                    continue

                if jj > kk:
                    txt = "\n".join(
                        [
                            rf"$\rho$={self.rho[kk,jj]:.2f}",
                            rf"$R^2$={self.r2[kk,jj]:.2f}",
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
            fig.savefig("similarity-matrix")
        else:
            fig.savefig(os.path.join(dest, "similarity-matrix"))

    @staticmethod
    def compute_iou(seta: list[float], setb: list[float]) -> float:
        """Compute the IOU of the value ranges spanned by two sets.

        Parameters
        ----------
        seta : list[float]
            Elements of set A.
        setb : list[float]
            Elements of set B.

        Returns
        -------
        float
            IOU of the ranges spanned by A and B.
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
    """Standardize a vector or time series to zero mean and unit variance.

    Parameters
    ----------
    x : np.ndarray
        The vector or time series to standardize.

    Returns
    -------
    np.ndarray
        The standardized vector or time series.
    """
    return (x - np.nanmean(x)) / np.nanstd(x)


def resample_to_n_samples(x: np.ndarray, n: int) -> np.ndarray:
    """Resample a vector or time series to a specified number of samples.

    Parameters
    ----------
    x : np.ndarray
        The input vector or time series to resample.
    n : int
        The desired number of samples in the output.

    Returns
    -------
    np.ndarray
        The resampled vector or time series with ``n`` samples.
    """
    original_length = len(x)
    if original_length == n:
        return x

    t = np.linspace(0, 1, n)
    tp = np.linspace(0, 1, original_length)
    return np.interp(t, tp, x)


def compare_signals(xs: np.ndarray, names: list[str]) -> ComparisonMetrics:
    """Compute pairwise comparison metrics for a set of signals.

    Parameters
    ----------
    xs : np.ndarray
        Signals to compare; each row is one signal.
    names : list[str]
        Names of the signals.

    Returns
    -------
    ComparisonMetrics
        Pairwise MAD, Pearson correlation, and R² matrices.
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
    """Plot direct scatter-plot comparisons between all pairs of signals.

    Parameters
    ----------
    xs : dict[str, np.ndarray]
        Dictionary of proxy signal arrays.
    metrics : ComparisonMetrics
        Pairwise comparison metrics.
    names : list[str]
        Case names.
    dest : str, optional
        Directory where the plot is saved. If ``None``, saves in the
        current working directory, by default None.
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
    """Mark FOG regions as shaded spans on a matplotlib axes object.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    flag : np.ndarray
        FOG flag array; positive transitions mark FOG onset.
    axs : pltlib.axes
        Axes on which to draw the FOG regions.
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
    """Overlay FI estimates from multiple methods on a single plot.

    FOG periods are highlighted with gray shading. The resulting figure
    is saved to disk.

    Parameters
    ----------
    t : np.ndarray
        1D array of recording time values.
    estimates : dict[str, np.ndarray]
        FI estimates keyed by method name; each value is a dictionary
        with ``"t"`` (time) and ``"fi"`` (freeze index) arrays.
    flag : np.ndarray
        1D boolean array indicating FOG presence at each time point.
    dest : str, optional
        Directory where the plot is saved. If ``None``, saves in the
        current working directory, by default None.
    standardized : bool, optional
        If ``True``, the filename and y-axis label reflect standardized
        FI values, by default False.

    Notes
    -----
    The figure is saved as ``fi-overlay-standardized`` or ``fi-overlay``
    depending on ``standardized``.
    """
    logger.info("Drawing ovelray comparison")
    YLABEL = "Standardized FI [-]" if standardized else "FI [-]"
    fn = "fi-overlay-standardized" if standardized else "fi-overlay"
    n = len(estimates)
    colors = cfg.generate_n_colors_from_cmap(n, cfg.COMP_CM)
    if "multitaper" in estimates.keys():
        colors[-1] = cfg.MT_COLOR

    colors = iter(colors)
    fig, axs = pltlib.subplots()
    mark_fog_regions_on_axs(t, flag, axs)
    for case, vals in estimates.items():
        kwargs = {
            "label": case.title(),
            "ls": "-",
            "lw": 3,
            "zorder": 5,
            "c": next(colors),
        }
        if case == "multitaper":
            kwargs.update({"lw": 2, "zorder": 10, "ls": "--"})
        elif case == "zach":
            kwargs.update({"zorder": 1})

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
    """Draw the power spectra of the FI estimates.

    Parameters
    ----------
    estimates : dict[str, np.ndarray]
        FI estimates keyed by method name; each value is a dictionary
        with ``"t"`` (time) and ``"fi"`` (freeze index) arrays.
    dest : str
        Directory where the plot is saved.
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
    colors = cfg.generate_n_colors_from_cmap(n, cfg.COMP_CM)
    if "Multitaper" in names:
        colors[-1] = cfg.MT_COLOR

    colors = iter(colors)
    fig, axs = pltlib.subplots()
    for f, x, name in zip(freqs, spectra, names):
        kwargs = {"label": name.title(), "ls": "-", "lw": 3, "c": next(colors)}
        if name == "Multitaper":
            kwargs.update({"lw": 3, "zorder": 10, "ls": "--"})

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
    """Compute metrics and generate comparison plots for a set of FI estimates.

    Parameters
    ----------
    t : np.ndarray
        Recording time array.
    estimates : dict[str, dict[str, np.ndarray]]
        FI estimates keyed by variant name, each containing ``"t"`` and
        ``"fi"`` arrays.
    dest : str
        Directory where plots are saved.
    flag : np.ndarray
        FOG flag signal array.
    standardized : bool, optional
        Whether the FI values are standardized, by default True.

    Returns
    -------
    comparison_metrics : ComparisonMetrics
        Pairwise comparison metrics for the provided FI estimates.
    names : list[str]
        Names of the compared FI variants.
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
    """Draw a sweep comparison plot for a single swept parameter.

    Parameters
    ----------
    t : np.ndarray
        Recording time array.
    param_values : np.ndarray
        Values of the swept parameter.
    estimates : list[dict[str, np.ndarray]]
        FI estimates for each parameter value; each entry contains
        ``"t"`` and ``"fi"`` arrays.
    param_name_label : tuple[str, str]
        Tuple of (file-name stem, axis label) for the swept parameter.
    dest : str, optional
        Directory where the plot is saved. If ``None``, saves in the
        current working directory, by default None.
    flags : np.ndarray, optional
        FOG flag signal used to shade FOG regions, by default None.
    standardized : bool, optional
        Whether the FI values are standardized, by default True.
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
    if flags is not None:
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
    else:
        axs.set_ylim((2, 10))

    fig.tight_layout()
    if dest is not None:
        fn = os.path.join(dest, fn)

    fig.savefig(fn)
    pltlib.close(fig)


def compute_and_visualize_ious(
    comparison: ComparisonMetrics, dest: str
) -> dict[str, list[float]]:
    """Compute and plot pairwise IOU metrics for all compared variants.

    Parameters
    ----------
    comparison : ComparisonMetrics
        Pairwise comparison metrics.
    dest : str
        Directory where plots are saved.

    Returns
    -------
    dict[str, list[float]]
        IOU values for MAD, rho, and R², keyed by metric name.
    """
    ious = {"mad": [], "rho": [], "r2": []}
    for name in comparison.names:
        mad, rho, r2 = comparison.compute_metrics_iou(name)
        ious["mad"].append(mad)
        ious["rho"].append(rho)
        ious["r2"].append(r2)

    markers = {"lumbar": "s", "thigh": "^", "shank": "o"}
    labels = {"mad": "IOU(MAD)", "rho": r"IOU($\rho$)", "r2": r"IOU($R^2$)"}

    if "multitaper" in [name.value for name in comparison.names]:
        colors = cfg.generate_n_colors_from_cmap(len(comparison.names), cfg.COMP_CM)
        colors[-1] = cfg.MT_COLOR
    else:
        colors = np.vstack(
            [
                cfg.generate_n_colors_from_cmap(
                    len(comparison.names) // len(markers), cfg.COMP_CM
                )
                for _ in range(len(markers))
            ]
        )

    n = len(comparison.names)
    figx = n if n < 7 else 10
    figy = n - 1 if n < 7 else 9
    for combo in itertools.combinations(ious.keys(), 2):
        a, b = combo
        fig, axs = pltlib.subplots(figsize=(figx, figy))
        for ii, name in enumerate(comparison.names):
            mk = name.split("-")[0]
            marker = markers[mk] if mk in markers.keys() else "o"
            if name == "multitaper":
                marker = "v"

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
