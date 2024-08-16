"""!
    Freezing Index Module
    =====================

    This module features functions for the computation of freeze indices
    such as the freezing index by Moore, the one by Bachlin, the one by
    Cockx, and the multitaper FI introduced by Magnes AG.

    @author A. Schaer
    @copyright Magnes AG, (C) 2024.
"""

import enum
import logging

import numpy as np
from scipy import signal
from scipy import fft


class FREQUENCY_RANGE(enum.Enum):
    LOCOMOTOR: tuple = (0.5, 3.0)
    FREEZING: tuple = (3.0, 8.0)


class FI_THS(enum.Enum):
    MOORE: float = 2.3
    ZACH: float = 1.47
    BACHLIN: float = 1.5


class VARIANTS(str, enum.Enum):
    MOORE: str = "moore"
    BACHLIN: str = "bachlin"
    COCKX: str = "cockx"
    ZACH: str = "zach"
    MULTITAPER: str = "multitaper"


MIN_FFT_WINDOW_SIZE = 256
HOP_DIV = 32

logger = logging.getLogger(__name__)


def compute_fi_variant(
    x: np.ndarray,
    fs: float = 100.0,
    variant: VARIANTS = VARIANTS.MULTITAPER,
    variant_kwargs: dict = {},
) -> tuple[np.ndarray]:
    """!Compute the FI on x using the selected variant

    @param x Proxy signal
    @param fs (default: 100.0) Sampling frequency in Hz
    @param variant (default: "multitaper") Variant to use for FI computation
    @param variant_kwargs (default: {}) Keyword arguments for the variant
    @return
    """
    if variant == VARIANTS.MOORE:
        return compute_moore_fi(x, fs)
    elif variant == VARIANTS.BACHLIN:
        return compute_bachlin_fi(x, fs)
    elif variant == VARIANTS.COCKX:
        return compute_cockx_fi(x, fs)
    elif variant == VARIANTS.MULTITAPER:
        return compute_multitaper_fi(x, fs, **variant_kwargs)
    else:
        raise ValueError(f"Unknown variant {variant}")


def compute_fi_free_window(
    x: np.ndarray,
    w: int,
    fs: float = 100.0,
) -> tuple[np.ndarray]:
    """!Compute the FI on x on a freely selectable window

    The FI definition introduced in
        Moore, S. T., MacDougall, H. G., & Ondo, W. G. (2008).
        Ambulatory monitoring of freezing of gait in Parkinson's disease.
        Journal of Neuroscience Methods, 167(2), 340-348.
        doi:10.1016/j.jneumeth.2007.08.023

    > A freeze index (FI) at time t was deﬁned as the square of the
        area under the power spectra of a 6 s window of data
        (centered at time t) in the 'freeze' band, divided
        by the square of the area under the spectra in the 'locomotor'
        band.

    that is the ratio of the signal energy in the "freezing"
    frequency band (3, 8) Hz to the "locomotor" frequency band (0.5, 3) Hz.

    This function detaches from the original definitoin by allowing the
    computation of the FI for any window size (duration), and not only for
    6 s.

    This function relies on a Hann window for tapering and uses constant detrending,
    both operations were not explicitly mentioned in the paper.

    @param x Gait raw signal
    @param w Window width in number of samples
    @param fs Sampling frequency, optional. Default 100 Hz
    @return t, FI
    """
    win = signal.windows.hann(w, sym=False)
    stf = signal.ShortTimeFFT(
        win,
        max(w // HOP_DIV, 1),
        fs,
        mfft=max(w, MIN_FFT_WINDOW_SIZE),
        scale_to="psd",
        fft_mode="onesided",
    )
    ghost = stf.spectrogram(x, detr="constant")
    locomotor_slc = slice(
        *[
            round(nf / (0.5 * fs) * ghost.shape[0])
            for nf in FREQUENCY_RANGE.LOCOMOTOR.value
        ]
    )
    freeze_slc = slice(
        *[
            round(nf / (0.5 * fs) * ghost.shape[0])
            for nf in FREQUENCY_RANGE.FREEZING.value
        ]
    )
    locomotor_ghost = ghost[locomotor_slc, :].copy()
    freeze_ghost = ghost[freeze_slc, :].copy()
    locomotor_f = np.linspace(
        *FREQUENCY_RANGE.LOCOMOTOR.value, locomotor_ghost.shape[0]
    )
    patho_f = np.linspace(*FREQUENCY_RANGE.FREEZING.value, freeze_ghost.shape[0])
    logger.debug(f"Locomotor frequency range: [{locomotor_f[0]}, {locomotor_f[-1]}]")
    logger.debug(f"Freezing frequency range: [{patho_f[0]}, {patho_f[-1]}]")
    locomotor_power = np.trapz(locomotor_ghost, locomotor_f, axis=0)
    freeze_power = np.trapz(freeze_ghost, patho_f, axis=0)
    fi = (freeze_power / (locomotor_power + np.spacing(1))) ** 2
    t = np.linspace(0, 1, len(fi))
    return t, fi


def apply_moore_fi_scaling(fi: np.ndarray) -> np.ndarray:
    """!Apply the FI scaling described in Moore et al

    "[...] multiplying by 100 and taking the natural logarithm."

    > Moore, S. T., MacDougall, H. G., & Ondo, W. G. (2008).
        Ambulatory monitoring of freezing of gait in Parkinson's disease.
        Journal of Neuroscience Methods, 167(2), 340-348.
        doi:10.1016/j.jneumeth.2007.08.023

    @param fi FI
    @return Scaled FI
    """
    return np.log(100 * fi)


def compute_moore_fi(proxy: np.ndarray, fs: float = 100.0) -> tuple[np.ndarray]:
    """!Compute the scaled FI according to Moore et al

    This function computes the FI for a window of 6 seconds and scales the
    value by the Moore scaling.

    In this implementation Tukey windows are used for tapering.

    @param proxy Proxy signal for FI computation. Originally: vertical acceleration.
    @param fs Signal sampling frequency
    @return t, Moore scaled FI
    """
    WINDOW_SURATION_S = 6.0
    w = int(WINDOW_SURATION_S * fs) + 1
    t, fi = compute_fi_free_window(proxy, w, fs)
    return t, apply_moore_fi_scaling(fi)


def compute_bachlin_fi(proxy: np.ndarray, fs: float = 100.0) -> tuple[np.ndarray]:
    """Compute the FI according to Bachlin et al

    This method is based on the MATLAB code provided with the Daphnet dataset.

    > Marc Bächlin, Meir Plotnik, Daniel Roggen, Inbal Maidan, Jeffrey M. Hausdorff,
        Nir Giladi, and Gerhard Tröster. Wearable Assistant for Parkinson's Disease Patients
        With the Freezing of Gait Symptom. IEEE Transactions on Information Technology
        in Biomedicine, 14(2), March 2010, pages 436-446

    @note The source declares a window size of 4 s. Given a sampling rate of 64 Hz in the
    original work, this corresponds to slices of 256 datapoints for the FFT evaluation.
    In this implementation, the window size is computed such that it matches the real-world
    duration (4 s) based on the provided sampling frequency. In the paper, it is said that
    windowing is done in steps of 0.5 seconds.

    Original MATLAB code (commented code and excessive blank lines removed)
    function res = x_fi(data,SR,stepSize)
        NFFT = 256;
        locoBand=[0.5 3];
        freezeBand=[3 8];
        windowLength=256;

        f_res = SR / NFFT;
        f_nr_LBs  = round(locoBand(1)   / f_res);
        f_nr_LBs( f_nr_LBs==0 ) = [];
        f_nr_LBe  = round(locoBand(2)   / f_res);
        f_nr_FBs  = round(freezeBand(1) / f_res);
        f_nr_FBe  = round(freezeBand(2) / f_res);
        d = NFFT/2;

        % Online implementation
        % jPos is the current position, 0-based, we take a window
        jPos = windowLength+1;      % This should not be +1 but we follow Baechlin's implementation.
        i=1;

        % Iterate the FFT windows
        while jPos <= length(data)
            jStart = jPos - windowLength + 1;
            % Time (sample nr) of this window
            time(i) = jPos;

            % get the signal in the window
            y = data(jStart:jPos);
            y = y - mean(y); % make signal zero-mean

            % Compute FFT
            Y = fft(y,NFFT);
            Pyy = Y.* conj(Y) / NFFT;


            % --- calculate sumLocoFreeze and freezeIndex ---
            areaLocoBand   = x_numericalIntegration( Pyy(f_nr_LBs:f_nr_LBe), SR );
            areaFreezeBand = x_numericalIntegration( Pyy(f_nr_FBs:f_nr_FBe),  SR );

            sumLocoFreeze(i) = areaFreezeBand + areaLocoBand;

            freezeIndex(i) = areaFreezeBand/areaLocoBand;
            % --------------------


            % next window
            jPos = jPos + stepSize;
            i = i + 1;
            %break;
        end

        res.sum = sumLocoFreeze;
        res.quot = freezeIndex;
        res.time = time;
    end

    Moore scaling is applied in this verison of the implementation.

    @param proxy Proxy signal, originally vertical acceleration
    @param fs Sampling frequency
    @return t, Bachlin FI
    """
    BACHLIN_WINDOW_DURATION_S = 4.0
    BACHLIN_STEP_DURATION_S = 0.5
    n = round(BACHLIN_WINDOW_DURATION_S * fs) + 1
    step = round(BACHLIN_STEP_DURATION_S * fs)
    fi = []
    f, loco_idx, freeze_idx = generate_freqs_locomotion_and_freeze_band_indices(n, fs)
    for center_idx in range(n // 2, len(proxy) - n // 2, step):
        x = proxy[center_idx - n // 2 : center_idx + 2 * (n + 1) // 4]
        if len(x) != n:
            logger.warning(f"Invalid window length detected. Is {len(x)}, expected {n}")
            continue

        X = fft.fft(x - np.mean(x))
        power_density = np.square(np.abs(X) / n)
        lpd = power_density[loco_idx]
        fpd = power_density[freeze_idx]
        lp = np.trapz(lpd, f[loco_idx])
        fp = np.trapz(fpd, f[freeze_idx])
        fi.append(fp / (lp + np.spacing(1)))

    logger.debug(f"Number of FI samples {len(fi)}")
    t = np.linspace(0, 1, len(fi) + 1)
    return t[1:], apply_moore_fi_scaling(np.array(fi))


def compute_cockx_fi(proxy: np.ndarray, fs: float = 100.0) -> tuple[np.ndarray]:
    """!Compute Cockx FI

    This is based on the MATLAB code provided with
    > Cockx H, Nonnekes J, Bloem BR, van Wezel R, Cameron I, Wang Y. Dealing with
        the heterogeneous presentations of freezing of gait: how reliable are the
        freezing index and heart rate for freezing detection?. Journal of
        neuroengineering and rehabilitation. 2023 Apr 27;20(1):53.

    The code was accessed from GitHub on August 07, 2024
        Repository https://github.com/helenacockx/FI-HR_duringFOG/
        FI computation https://github.com/helenacockx/FI-HR_duringFOG/blob/main/D.data_preprocessing/calculate_FI.m

    From the code and paper following is inferred:
    * A window size of 3 seconds was used
    * An overlap of W/2 was used
    * Hann windows are applied to the windows, without any additional detrending step
    * PSD computed as squared FFT magnitude

    @param proxy Proxy signal from where to derive FI, originally shin vertical acceleration
    @param fs Sampling frequency
    @return t, FI according to Cockx
    """
    COCKX_WINDOW_DURATION_S = 3.0
    n = round(COCKX_WINDOW_DURATION_S * fs) + 1
    fi = []
    f, loco_idx, freeze_idx = generate_freqs_locomotion_and_freeze_band_indices(n, fs)
    w = signal.windows.hann(n)
    for center_idx in range(n // 2, len(proxy) - n // 2, n // 2):
        x = proxy[center_idx - n // 2 : center_idx + 2 * (n + 1) // 4]
        if len(x) != n:
            logger.warning(f"Invalid window length detected. Is {len(x)}, expected {n}")
            continue

        X = fft.fft(x * w)
        power_density = np.square(np.abs(X)) / n
        lpd = power_density[loco_idx]
        fpd = power_density[freeze_idx]
        lp = np.trapz(lpd, f[loco_idx])
        fp = np.trapz(fpd, f[freeze_idx])
        fi.append((fp / (lp + np.spacing(1))) ** 2)

    t = np.linspace(0, 1, len(fi) + 1)
    return t[1:], apply_moore_fi_scaling(np.array(fi))


def compute_zach_fi(proxy: np.ndarray, fs: float = 100.0) -> tuple[np.ndarray]:
    """Compute FI according to Zach

    This is a Moore FI with window size set to 2 s.
    """
    ZACH_WINDOW_DURATION_S = 2.0
    w = int(fs * ZACH_WINDOW_DURATION_S) + 1
    t, fi = compute_fi_free_window(proxy, w, fs)
    return t, apply_moore_fi_scaling(fi)


def compute_multitaper_fi(
    proxy: np.ndarray, fs: float = 100.0, dt: float = 5.0, L: int = 4, NW: float = 2.5
) -> tuple[np.ndarray]:
    """!Compute multitaper FI with L DPSS-tapers

    This is based on the multi-taper spectral estimation algorithm described in
    > Babadi B, Brown EN. A review of multitaper spectral analysis. IEEE Transactions
        on Biomedical Engineering. 2014 Mar 14;61(5):1555-64.

    @param proxy Proxy signal
    @param fs Sampling frequency
    @param dt Time window
    @param L Number of tapers
    @param NW DPSS half-bandwidth parameter. 2.5 defualt
    @return t, FI
    """
    n = round(dt * fs) + 1
    nfft = 2 ** int(np.log2(n) + 3)
    windows = signal.windows.dpss(n, NW, L, sym=False)
    fi = []
    f, loco_idx, freeze_idx = generate_freqs_locomotion_and_freeze_band_indices(
        nfft, fs
    )

    for center_idx in range(n // 2, len(proxy) - n // 2, HOP_DIV):
        spectrum = np.zeros(nfft)
        x = proxy[center_idx - n // 2 : center_idx + 2 * (n + 1) // 4]
        if len(x) != n:
            logger.warning(f"Invalid window length detected. Is {len(x)}, expected {n}")
            continue

        y = signal.detrend(x)
        for w in windows:
            Y = fft.fft(y * w, n=nfft)
            spectrum += np.square(np.abs(Y))

        lpd = spectrum[loco_idx]
        fpd = spectrum[freeze_idx]
        lp = np.trapz(lpd, f[loco_idx])
        fp = np.trapz(fpd, f[freeze_idx])
        fi.append(fp / (lp + np.spacing(1)))

    fi = np.array(fi)
    nmaf = 5
    mafk = np.ones(nmaf) / nmaf
    fi = np.convolve(fi, mafk, mode="same")

    t = np.linspace(0, 1, len(fi) + 1)
    return t[1:], apply_moore_fi_scaling(fi)


def generate_freqs_locomotion_and_freeze_band_indices(
    nfft: int, fs: float
) -> tuple[np.ndarray]:
    """!Generate frequencies, locomotion-band indices, and freezing-band indices

    @param nfft FFT samples
    @param fs Sampling frequency
    @return FFT frequency, locomotion index mask, freezing index mask
    """
    f = fft.fftfreq(nfft, d=1.0 / fs)
    loco_idx = (f >= FREQUENCY_RANGE.LOCOMOTOR.value[0]) * (
        f <= FREQUENCY_RANGE.LOCOMOTOR.value[1]
    )
    freeze_idx = (f <= FREQUENCY_RANGE.FREEZING.value[1]) * (
        f >= FREQUENCY_RANGE.FREEZING.value[0]
    )
    return f, loco_idx, freeze_idx


def compute_babadi_brown_multitaper_fi(
    proxy: np.ndarray,
    fs: float = 100.0,
    dt: float = 1.0,
    spectral_resolution: float = 0.05,
) -> tuple[np.ndarray]:
    """!Compute multitaper FI based on multi-tapers spectral estimaition according to Babadi-Brown

    The number of tapers are defined based on the multi-taper spectral estimation
    algorithm described in
    > Babadi B, Brown EN. A review of multitaper spectral analysis. IEEE Transactions
        on Biomedical Engineering. 2014 Mar 14;61(5):1555-64.

    @param proxy Proxy signal
    @param fs Sampling frequency
    @param dt Time window
    @return t, FI
    """
    n = int(dt * fs)
    alpha = 0.5 * n * dt * spectral_resolution
    if alpha < 1:
        raise ValueError(
            "alpha must be greater than one: increase horizon (time window `dt`) "
            "for the given sampling frequency, or decrease target resolution (larger value)."
        )

    L = int(2 * alpha) - 1
    NW = 2.5
    logger.debug(f"Number of tapers: {L}")
    return compute_multitaper_fi(proxy, fs, dt, L, NW)


def combine_fis(
    lt: list[float], lfi: list[float], rt: list[float], rfi: list[float]
) -> tuple[np.ndarray]:
    """!Combine FI sequences

    It is implicitly assumed that the two FI sequences correspond to the same
    time window. It is further assumed that the FI sequences feature equispaced
    samples. This funciton combines the two FI sequences by resampling them
    to have them feature the same number of samples and then taking the maximum
    at each sample.

    @param fi_a First FI array
    @param fi_b Second FI array
    @return
    """

    t = np.linspace(min(lt[0], rt[0]), max(lt[-1], rt[-1]), np.lcm(len(lt), len(rt)))
    return t.tolist(), np.max(
        np.vstack([np.interp(t, lt, lfi), np.interp(t, rt, rfi)]), axis=0
    )
