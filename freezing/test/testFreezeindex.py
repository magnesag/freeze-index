"""!
    Test the Freeze Index Module
    ============================

    @author A. Schaer
    @copyright Magnes AG, (C) 2024.
"""

import os

import sys
import unittest as ut

import matplotlib.pyplot as pltlib
import numpy as np


FILE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.join(FILE_DIR, "..", "..")
RES_DIR = os.path.join(FILE_DIR, "res")
PLT_RC = {
    "figure": {"figsize": (10, 5)},
    "font": {"family": "arial", "size": 16},
    "savefig": {"format": "png", "dpi": 300},
    "text": {"usetex": False},
}


sys.path.append(ROOT)

import freezing.freezeindex as frz

for kk, vv in PLT_RC.items():
    pltlib.rc(kk, **vv)

if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)


class TestFreezingFunctions(ut.TestCase):
    def setUp(self):
        self.fs = 100
        t1 = 30
        n = int(t1 * self.fs) + 1
        self.t = np.linspace(0, t1, n)
        self.x = np.ones_like(self.t) * 9.81
        flow = 1.2
        fhigh = 5
        motion = np.sin(flow * 2 * np.pi * self.t)
        self.t_start_freeze = 10.0
        self.t_end_freeze = 20.0
        idx = (self.t >= self.t_start_freeze) * (self.t < self.t_end_freeze)
        motion[idx] += np.sin(fhigh * 2 * np.pi * self.t[idx])
        self.x += motion

    def tearDown(self):
        pltlib.close("all")

    @staticmethod
    def standardize(x):
        return (x - np.nanmean(x)) / np.nanstd(x)

    def mark_freezing_region_on_axes(self, axs: pltlib.Axes):
        axs.axvspan(
            self.t_start_freeze,
            self.t_end_freeze,
            0,
            1,
            fc="gray",
            alpha=0.5,
            label="Freezing Region",
        )

    def test_proxy_signal(self):
        fig, axs = pltlib.subplots()
        axs.plot(self.t, self.x, c="black")
        self.mark_freezing_region_on_axes(axs)
        axs.grid(True)
        axs.legend(loc="upper right")
        axs.set(xlabel="Time [s]", ylabel="Proxy [a.u.]")
        fig.tight_layout()
        fig.savefig(os.path.join(RES_DIR, "proxy-signal"))

    def test_generate_freqs_locomotion_and_freeze_band_indices(self):
        nfft = 128
        fs = 128.0
        cf, cli, cfi = frz.generate_freqs_locomotion_and_freeze_band_indices(nfft, fs)

        self.assertEqual(nfft, len(cf))
        self.assertTrue(cf[cli][0] >= frz.FREQUENCY_RANGE.LOCOMOTOR.value[0])
        self.assertTrue(cf[cli][-1] <= frz.FREQUENCY_RANGE.LOCOMOTOR.value[1])
        self.assertTrue(cf[cfi][0] >= frz.FREQUENCY_RANGE.FREEZING.value[0])
        self.assertTrue(cf[cfi][-1] <= frz.FREQUENCY_RANGE.FREEZING.value[1])

        k = np.arange(nfft)
        fig, axs = pltlib.subplots()
        pltlib.axvspan(
            *frz.FREQUENCY_RANGE.LOCOMOTOR.value,
            fc="blue",
            label="Locomotor band",
            alpha=0.2,
        )
        pltlib.axvspan(
            *frz.FREQUENCY_RANGE.FREEZING.value,
            fc="red",
            label="Freezing band",
            alpha=0.2,
        )
        axs.plot(k, cf, c="black")
        axs.plot(k[cli], cf[cli], c="blue", label="Locomotor frequencies", ls="--")
        axs.plot(k[cfi], cf[cfi], c="red", label="Freezing frequencies", ls="--")
        axs.set(xlabel="Sample [-]", ylabel="FFT frequency [Hz]", xlim=(0, nfft / 2))
        axs.grid(True)
        axs.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(RES_DIR, "frequency-bands"))

    def test_compute_fi_free_window(self):
        test_wins = [ii * int(self.fs) for ii in range(1, 7)]
        fis = [frz.compute_fi_free_window(self.x, w, self.fs) for w in test_wins]
        fig, axs = pltlib.subplots()
        self.mark_freezing_region_on_axes(axs)
        for w, (t, fi) in zip(test_wins, fis):
            axs.plot(
                t * (self.t[-1] - self.t[0]) + self.t[0],
                self.standardize(frz.apply_moore_fi_scaling(fi)),
                label=f"W = {w}",
            )

        axs.grid(True)
        axs.set(xlabel="Time [s]", ylabel="FI [-]")
        axs.legend(loc="upper left", bbox_to_anchor=(1, 1))
        fig.tight_layout()
        fig.savefig(os.path.join(RES_DIR, "free-windows-fi"))

    def test_compute_moore_freezing_index(self):
        ct, cfi = frz.compute_moore_fi(self.x, self.fs)
        ct = ct * (self.t[-1] - self.t[0]) + self.t[0]
        indicator = np.zeros_like(ct)
        indicator[cfi > frz.FI_THS.MOORE.value] = 1.0
        # Remove false positive at end and start due to edge effects
        indicator[9 * len(indicator) // 10 :] = 0.0
        indicator[: 2 * len(indicator) // 10] = 0.0
        di = np.diff(indicator)
        indicator_start = ct[np.arange(len(indicator) - 1)[di > 0]][0]
        indicator_end = ct[np.arange(len(indicator) - 1)[di < 0]][0]

        union_start = min(self.t_start_freeze, indicator_start)
        union_end = max(self.t_end_freeze, indicator_end)
        union = union_end - union_start
        intersection_start = max(self.t_start_freeze, indicator_start)
        intersection_end = min(self.t_end_freeze, indicator_end)
        intersection = max(0, intersection_end - intersection_start)
        iou = intersection / union
        print(f"IOU = {iou:.2f}", end=" ... ", flush=True)

        fig, axs = pltlib.subplots()
        self.mark_freezing_region_on_axes(axs)
        axs.plot(ct, cfi, c="black")
        axs.plot(ct, indicator, c="gray")
        axs.axvline(intersection_start, color="blue", ls="--")
        axs.axvline(intersection_end, color="blue", ls="--")
        axs.axvline(union_start, color="orange", ls="--")
        axs.axvline(union_end, color="orange", ls="--")
        axs.axhline(frz.FI_THS.MOORE.value, ls="--", color="red")
        axs.grid(True)
        axs.set(xlabel="Time [s]", ylabel="Moore FI [-]", title=f"IOU = {iou:.2f}")
        fig.tight_layout()
        fig.savefig(os.path.join(RES_DIR, "moore-fi"))

        self.assertTrue(iou > 0.5)

    def test_compute_bachlin_fi(self):
        ct, cfi = frz.compute_bachlin_fi(self.x, self.fs)
        ct = ct * (self.t[-1] - self.t[0]) + self.t[0]

        fig, axs = pltlib.subplots()
        self.mark_freezing_region_on_axes(axs)
        axs.plot(ct, cfi, c="black")
        axs.grid(True)
        axs.set(xlabel="Time [s]", ylabel="Bachlin FI [-]")
        fig.tight_layout()
        fig.savefig(os.path.join(RES_DIR, "bachlin-fi"))

    def test_compute_cockx_fi(self):
        ct, cfi = frz.compute_cockx_fi(self.x, self.fs)
        ct = ct * (self.t[-1] - self.t[0]) + self.t[0]

        fig, axs = pltlib.subplots()
        self.mark_freezing_region_on_axes(axs)
        axs.plot(ct, cfi, c="black")
        axs.grid(True)
        axs.set(xlabel="Time [s]", ylabel="Cockx FI [-]")
        fig.tight_layout()
        fig.savefig(os.path.join(RES_DIR, "cockx-fi"))

    def test_compute_zach_fi(self):
        ct, cfi = frz.compute_zach_fi(self.x, self.fs)
        ct = ct * (self.t[-1] - self.t[0]) + self.t[0]

        fig, axs = pltlib.subplots()
        self.mark_freezing_region_on_axes(axs)
        axs.plot(ct, cfi, c="black")
        axs.grid(True)
        axs.set(xlabel="Time [s]", ylabel="Zach FI [-]")
        fig.tight_layout()
        fig.savefig(os.path.join(RES_DIR, "zach-fi"))

    def test_compute_multitaper_fi(self):
        ct, cfi = frz.compute_multitaper_fi(self.x, self.fs, dt=5.0, L=4, NW=2.5)
        ct = ct * (self.t[-1] - self.t[0]) + self.t[0]

        fig, axs = pltlib.subplots()
        self.mark_freezing_region_on_axes(axs)
        axs.plot(ct, cfi, c="black")
        axs.grid(True)
        axs.set(xlabel="Time [s]", ylabel="Multitaper FI [-]")
        fig.tight_layout()
        fig.savefig(os.path.join(RES_DIR, "multitaper-fi"))

    def test_compare_fi_implementations(self):
        tmoore, fimoore = frz.compute_moore_fi(self.x, self.fs)
        tbachlin, fibachlin = frz.compute_bachlin_fi(self.x, self.fs)
        tcockx, ficockx = frz.compute_cockx_fi(self.x, self.fs)
        tzach, fizach = frz.compute_zach_fi(self.x, self.fs)
        tmt, fimt = frz.compute_multitaper_fi(self.x, self.fs, dt=5.0, L=4, NW=2.5)

        dt = self.t[-1] - self.t[0]
        fig, axs = pltlib.subplots()
        self.mark_freezing_region_on_axes(axs)
        axs.plot(tmoore * dt + self.t[0], self.standardize(fimoore), label="Moore")
        axs.plot(
            tbachlin * dt + self.t[0], self.standardize(fibachlin), label="Bachlin"
        )
        axs.plot(tcockx * dt + self.t[0], self.standardize(ficockx), label="Cockx")
        axs.plot(tzach * dt + self.t[0], self.standardize(fizach), label="Zach")
        axs.plot(tmt * dt + self.t[0], self.standardize(fimt), label="Multi-taper")
        axs.grid(True)
        axs.legend(loc="upper left", bbox_to_anchor=(1, 1))
        axs.set(xlabel="Scaled time [a.u.]", ylabel="FI [-]")
        fig.tight_layout()
        fig.savefig(os.path.join(RES_DIR, "fi-comparison"))

    def test_combine_fis(self):
        t_left, fi_left = frz.compute_moore_fi(self.x, self.fs)
        t_right = np.linspace(0, 1, len(t_left) + 10)
        fi_right = np.interp(t_right, t_left, fi_left)
        ct, cfi = frz.combine_fis(t_left, fi_left, t_right, fi_right)

        fig, axs = pltlib.subplots()
        axs.plot(ct, cfi, lw=3, c="black", label="combo")
        axs.plot(t_left, fi_left, "--", label="left")
        axs.plot(t_right, fi_right, "--", label="right")
        axs.grid()
        axs.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(RES_DIR, "fi-combination"))

    def test_compare_fi_implementations_corrupted_sweeping(self):
        fs = 100
        t1 = 60
        n = int(fs * t1) + 1
        np.random.seed(0)
        t = np.linspace(0, t1, n)
        f = 8 * np.sin(np.pi * t / t1) ** 2
        x = np.sin(2 * np.pi * f * t) + 1.0 * np.random.randn(n)
        tmoore, fimoore = frz.compute_moore_fi(x, fs)
        tbachlin, fibachlin = frz.compute_bachlin_fi(x, fs)
        tcockx, ficockx = frz.compute_cockx_fi(x, fs)
        tzach, fizach = frz.compute_zach_fi(x, fs)
        tmt, fimt = frz.compute_multitaper_fi(x, fs, dt=5, L=4, NW=2.5)
        dt = t[-1] - t[0]

        fig, axs = pltlib.subplots(nrows=2, sharex=True)
        axs[0].plot(t, x, c="black")
        axs[0].plot(t, f, c="red")
        axs[0].set(ylabel="Frequency [Hz]")
        axs[1].plot(tmoore * dt + t[0], self.standardize(fimoore), label="Moore", lw=2)
        axs[1].plot(
            tbachlin * dt + t[0], self.standardize(fibachlin), label="Bachlin", lw=2
        )
        axs[1].plot(tcockx * dt + t[0], self.standardize(ficockx), label="Cockx", lw=2)
        axs[1].plot(tzach * dt + t[0], self.standardize(fizach), label="Zach", lw=2)
        axs[1].plot(
            tmt * dt + t[0], self.standardize(fimt), label="Multitaper", c="black", lw=3
        )
        axs[1].grid(True)
        axs[1].legend(loc="lower left", bbox_to_anchor=(0, 1), ncols=len(frz.VARIANTS))
        axs[1].set(
            xlabel="Time [s]",
            ylabel="Standardized FI [-]",
            xlim=(0, t1),
            ylim=(-2.5, 2.5),
        )
        fig.tight_layout()
        fig.savefig(os.path.join(RES_DIR, "fi-comparison-sweeping-seq"))

    def test_compare_fi_implementations_pure_random_sequence(self):
        fs = 100
        t1 = 60
        n = int(fs * t1) + 1
        np.random.seed(0)
        t = np.linspace(0, t1, n)
        x = 10.0 * np.random.randn(n)
        tmoore, fimoore = frz.compute_moore_fi(x, fs)
        tbachlin, fibachlin = frz.compute_bachlin_fi(x, fs)
        tcockx, ficockx = frz.compute_cockx_fi(x, fs)
        tzach, fizach = frz.compute_zach_fi(x, fs)
        tmt, fimt = frz.compute_multitaper_fi(x, fs, dt=5, L=4, NW=2.5)
        dt = t[-1] - t[0]

        fig, axs = pltlib.subplots()
        axs.plot(tmoore * dt + t[0], self.standardize(fimoore), label="Moore", lw=2)
        axs.plot(
            tbachlin * dt + t[0], self.standardize(fibachlin), label="Bachlin", lw=2
        )
        axs.plot(tcockx * dt + t[0], self.standardize(ficockx), label="Cockx", lw=2)
        axs.plot(tzach * dt + t[0], self.standardize(fizach), label="Zach", lw=2)
        axs.plot(
            tmt * dt + t[0], self.standardize(fimt), label="Multitaper", c="black", lw=3
        )
        axs.grid(True)
        axs.legend(loc="lower left", bbox_to_anchor=(0, 1), ncols=len(frz.VARIANTS))
        axs.set(
            xlabel="Time [s]",
            ylabel="Standardized FI [-]",
            xlim=(0, t1),
            ylim=(-2.5, 2.5),
        )
        fig.tight_layout()
        fig.savefig(os.path.join(RES_DIR, "fi-comparison-random-seq"))

    def test_compute_fi_variant_by_name(self):
        er = frz.compute_multitaper_fi(self.x, self.fs)

        # Test access via string
        cr = frz.compute_fi_variant(self.x, self.fs, variant="multitaper")
        np.testing.assert_array_almost_equal(cr[0], er[0])
        np.testing.assert_array_almost_equal(cr[1], er[1])

        # Test access via enum
        cr = frz.compute_fi_variant(self.x, self.fs, variant=frz.VARIANTS.MULTITAPER)
        np.testing.assert_array_almost_equal(cr[0], er[0])
        np.testing.assert_array_almost_equal(cr[1], er[1])
