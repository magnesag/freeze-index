"""
Freezing Index Module
=====================

This module features functions for the computation of freeze indices
such as the freezing index by Moore, the one by Bachlin, the one by
Cockx, and the multitaper FI introduced by Magnes AG.

If you use this code for any scientific publication, please cite
https://doi.org/10.3389/fneur.2025.1528963.

Authors:
    - A. Schaer
    - C. Mangiante
    - R. Sobkuliak
    - H. Maurenbrecher
Copyright:
    Magnes AG, (C) 2024.
"""

import enum
import logging

import numpy as np
from scipy import signal
from scipy import fft
from scipy import integrate


class FREQUENCY_RANGE(tuple, enum.Enum):
    """Frequency ranges for FI band power integration.

    Attributes
    ----------
    LOCOMOTOR : tuple
        Locomotor frequency band, (0.5, 3.0) Hz.
    FREEZING : tuple
        Freezing frequency band, (3.0, 8.0) Hz.
    FREEZING_COCKX : tuple
        Freezing frequency band per Cockx et al., (3.5, 8.0) Hz.
    """

    LOCOMOTOR = (0.5, 3.0)
    FREEZING = (3.0, 8.0)
    FREEZING_COCKX = (3.5, 8.0)


class FI_THS(float, enum.Enum):
    """Freeze index classification thresholds.

    Attributes
    ----------
    MOORE : float
        Threshold per Moore et al., 2.3.
    ZACH : float
        Threshold per Zach et al., 1.47.
    BACHLIN : float
        Threshold per Bachlin et al., 1.5.
    """

    MOORE = 2.3
    ZACH = 1.47
    BACHLIN = 1.5


class VARIANTS(str, enum.Enum):
    """Available freeze index computation variants.

    Attributes
    ----------
    MOORE : str
        Moore FI variant.
    BACHLIN : str
        Bachlin FI variant.
    COCKX : str
        Cockx FI variant.
    ZACH : str
        Zach FI variant.
    MULTITAPER : str
        Multitaper FI variant (Magnes AG).
    """

    MOORE = "moore"
    BACHLIN = "bachlin"
    COCKX = "cockx"
    ZACH = "zach"
    MULTITAPER = "multitaper"


# NOTE the minimum FFT window size is set such that for the Daphnet data (sampled at 64 Hz)
# and the Zach method (window size of 2 seconds), no (artificial) padding is added.
# This is meant to mimimize deviations from the original definitions.
# It should be noted though, that increasing this value will start to show the importance of
# adequate preprocessing steps in the spectrogram evaluation.
MIN_FFT_WINDOW_SIZE = 256
HOP_DIV = 32

logger = logging.getLogger(__name__)


def compute_fi_variant(
    x: np.ndarray,
    fs: float = 100.0,
    variant: VARIANTS = VARIANTS.MULTITAPER,
    variant_kwargs: dict = {},
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the FI on x using the selected variant.

    Parameters
    ----------
    x : np.ndarray
        Proxy signal.
    fs : float, optional
        Sampling frequency in Hz, by default 100.0.
    variant : VARIANTS, optional
        Variant to use for FI computation, by default VARIANTS.MULTITAPER.
    variant_kwargs : dict, optional
        Keyword arguments for the variant, by default {}.

    Returns
    -------
    t : np.ndarray
        Normalized time array.
    fi : np.ndarray
        Freeze index values.
    """
    if variant == VARIANTS.MOORE:
        return compute_moore_fi(x, fs)
    elif variant == VARIANTS.BACHLIN:
        return compute_bachlin_fi(x, fs)
    elif variant == VARIANTS.COCKX:
        return compute_cockx_fi(x, fs)
    elif variant == VARIANTS.ZACH:
        return compute_zach_fi(x, fs)
    elif variant == VARIANTS.MULTITAPER:
        return compute_multitaper_fi(x, fs, **variant_kwargs)
    else:
        raise ValueError(f"Unknown variant {variant}")


def compute_fi_free_window(
    x: np.ndarray,
    w: int,
    fs: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the FI on x on a freely selectable window.

    The FI definition introduced in [1]_:

        A freeze index (FI) at time t was defined as the square of the
        area under the power spectra of a 6 s window of data
        (centered at time t) in the 'freeze' band, divided by the square
        of the area under the spectra in the 'locomotor' band.

    That is the ratio of the signal power in the "freezing" frequency
    band (3, 8) Hz to the "locomotor" frequency band (0.5, 3) Hz.

    This function detaches from the original definition by allowing the
    computation of the FI for any window size (duration), and not only
    for 6 s.

    The follow-up paper by Moore et al. (2013) changed the locomotor band
    to (0, 3) Hz and the window size to 5 s. This variant is not
    implemented herein.

    Parameters
    ----------
    x : np.ndarray
        Gait raw signal.
    w : int
        Window width in number of samples.
    fs : float, optional
        Sampling frequency in Hz, by default 100.0.

    Returns
    -------
    t : np.ndarray
        Normalized time array.
    fi : np.ndarray
        Freeze index values.

    References
    ----------
    .. [1] Moore, S. T., MacDougall, H. G., & Ondo, W. G. (2008).
       Ambulatory monitoring of freezing of gait in Parkinson's disease.
       Journal of Neuroscience Methods, 167(2), 340-348.
       doi:10.1016/j.jneumeth.2007.08.023
    """
    win = signal.windows.boxcar(w, sym=False)
    stf = signal.ShortTimeFFT(
        win,
        max(w // HOP_DIV, 1),
        fs,
        mfft=max(w, MIN_FFT_WINDOW_SIZE),
        scale_to="psd",
        fft_mode="onesided",
    )
    ghost = stf.spectrogram(x, detr=None)
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
    locomotor_power = integrate.trapezoid(locomotor_ghost, locomotor_f, axis=0)
    freeze_power = integrate.trapezoid(freeze_ghost, patho_f, axis=0)
    fi = (freeze_power / (locomotor_power + np.spacing(1))) ** 2
    t = stf.t(len(x))
    t0 = 0
    t1 = (len(x) - 1) / fs
    idx = (t >= t0) * (t < t1)
    fi = fi[idx]
    return np.linspace(0, 1, len(fi)), fi


def apply_moore_fi_scaling(fi: np.ndarray) -> np.ndarray:
    """Apply the FI scaling described in Moore et al.

    Scaling is defined as "[...] multiplying by 100 and taking the natural
    logarithm." [1]_.

    Parameters
    ----------
    fi : np.ndarray
        Freeze index values.

    Returns
    -------
    np.ndarray
        Scaled freeze index values.

    References
    ----------
    .. [1] Moore, S. T., MacDougall, H. G., & Ondo, W. G. (2008).
       Ambulatory monitoring of freezing of gait in Parkinson's disease.
       Journal of Neuroscience Methods, 167(2), 340-348.
       doi:10.1016/j.jneumeth.2007.08.023
    """
    return np.log(100 * fi)


def compute_moore_fi(
    proxy: np.ndarray, fs: float = 100.0
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the scaled FI according to Moore et al.

    Computes the FI for a window of 6 seconds and scales the result by
    the Moore scaling as described in [1]_.

    The follow-up paper by Moore et al. (2013) changed the locomotor band
    to (0, 3) Hz and the window size to 5 s. This variant is not
    implemented herein.

    Parameters
    ----------
    proxy : np.ndarray
        Proxy signal for FI computation. Originally: vertical acceleration.
    fs : float, optional
        Signal sampling frequency in Hz, by default 100.0.

    Returns
    -------
    t : np.ndarray
        Normalized time array.
    fi : np.ndarray
        Moore-scaled freeze index values.

    References
    ----------
    .. [1] Moore, S. T., MacDougall, H. G., & Ondo, W. G. (2008).
       Ambulatory monitoring of freezing of gait in Parkinson's disease.
       Journal of Neuroscience Methods, 167(2), 340-348.
       doi:10.1016/j.jneumeth.2007.08.023
    """
    WINDOW_DURATION_S = 6.0
    w = int(WINDOW_DURATION_S * fs) + 1
    t, fi = compute_fi_free_window(proxy, w, fs)
    return t, apply_moore_fi_scaling(fi)


def compute_bachlin_fi(
    proxy: np.ndarray, fs: float = 100.0
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the FI according to Bachlin et al.

    Based on the MATLAB code provided with the Daphnet dataset [1]_.

    The source declares a window size of 4 s. Given a sampling rate of
    64 Hz in the original work, this corresponds to slices of 256
    datapoints for the FFT evaluation. In this implementation the window
    size is computed to match the real-world duration (4 s) based on the
    provided sampling frequency. Windowing is done in steps of 0.5 s
    as stated in the paper.

    Original MATLAB code (commented code and excessive blank lines
    removed):

    .. code-block:: matlab

        function res = x_fi(data,SR,stepSize)
            NFFT = 256;
            locoBand = [0.5 3];
            freezeBand = [3 8];
            windowLength = 256;

            f_res = SR / NFFT;
            f_nr_LBs  = round(locoBand(1) / f_res);
            f_nr_LBs(f_nr_LBs==0) = [];
            f_nr_LBe = round(locoBand(2) / f_res);
            f_nr_FBs = round(freezeBand(1) / f_res);
            f_nr_FBe = round(freezeBand(2) / f_res);
            d = NFFT / 2;

            % Online implementation
            % jPos is the current position, 0-based, we take a window
            jPos = windowLength + 1;
            i = 1;

            % Iterate the FFT windows
            while jPos <= length(data)
                jStart = jPos - windowLength + 1;
                % Time (sample nr) of this window
                time(i) = jPos;

                % get the signal in the window
                y = data(jStart:jPos);
                y = y - mean(y); % make signal zero-mean

                % Compute FFT
                Y = fft(y, NFFT);
                Pyy = Y.* conj(Y) / NFFT;

                % --- calculate sumLocoFreeze and freezeIndex ---
                areaLocoBand   = x_numericalIntegration(Pyy(f_nr_LBs:f_nr_LBe), SR);
                areaFreezeBand = x_numericalIntegration(Pyy(f_nr_FBs:f_nr_FBe),  SR);

                sumLocoFreeze(i) = areaFreezeBand + areaLocoBand;

                freezeIndex(i) = areaFreezeBand/areaLocoBand;
                % --------------------
                % next window
                jPos = jPos + stepSize;
                i = i + 1;
            end

            res.sum = sumLocoFreeze;
            res.quot = freezeIndex;
            res.time = time;
        end

    Moore scaling is applied in this version of the implementation.

    Parameters
    ----------
    proxy : np.ndarray
        Proxy signal, originally vertical acceleration.
    fs : float, optional
        Sampling frequency in Hz, by default 100.0.

    Returns
    -------
    t : np.ndarray
        Normalized time array.
    fi : np.ndarray
        Bachlin freeze index values.

    References
    ----------
    .. [1] Marc Bächlin, Meir Plotnik, Daniel Roggen, Inbal Maidan,
       Jeffrey M. Hausdorff, Nir Giladi, and Gerhard Tröster. Wearable
       Assistant for Parkinson's Disease Patients With the Freezing of
       Gait Symptom. IEEE Transactions on Information Technology in
       Biomedicine, 14(2), March 2010, pages 436-446.
    """
    BACHLIN_WINDOW_DURATION_S = 4.0
    BACHLIN_STEP_DURATION_S = 0.5
    n = round(BACHLIN_WINDOW_DURATION_S * fs) + 1
    step = round(BACHLIN_STEP_DURATION_S * fs)
    n_over_step = n // step
    fi = []
    f, loco_idx, freeze_idx = generate_freqs_locomotion_and_freeze_band_indices(n, fs)
    for center_idx in range(n // 2, len(proxy) - n // 2, step):
        x = proxy[center_idx - n // 2 : center_idx + 2 * (n + 1) // 4]
        if len(x) != n:
            logger.warning(f"Invalid window length detected. Is {len(x)}, expected {n}")
            continue

        X = fft.fft(x - np.mean(x))
        power_density = np.square(np.abs(X)) / n
        lpd = power_density[loco_idx]
        fpd = power_density[freeze_idx]
        lp = integrate.trapezoid(lpd, f[loco_idx])
        fp = integrate.trapezoid(fpd, f[freeze_idx])
        fi.append(fp / (lp + np.spacing(1)))

    logger.debug(f"Number of FI samples {len(fi)}")
    t = np.linspace(0, 1, len(fi) + n_over_step)
    return t[n_over_step // 2 : -n_over_step // 2], np.array(fi)


def compute_cockx_fi(
    proxy: np.ndarray, fs: float = 100.0
) -> tuple[np.ndarray, np.ndarray]:
    """Compute FI according to Cockx et al.

    Based on the MATLAB code provided with [1]_, accessed from GitHub on
    August 07, 2024 (repository
    https://github.com/helenacockx/FI-HR_duringFOG/).

    From the code and paper the following is inferred:

    - A window size of 3 seconds was used.
    - An overlap of W/2 was used.
    - Hann windows are applied without any additional detrending step.
    - PSD is computed as squared FFT magnitude.
    - The freezing band per the paper is (3.5, 8) Hz, although the
      MATLAB code uses (3, 8) Hz.

    Parameters
    ----------
    proxy : np.ndarray
        Proxy signal from where to derive FI, originally shin vertical
        acceleration.
    fs : float, optional
        Sampling frequency in Hz, by default 100.0.

    Returns
    -------
    t : np.ndarray
        Normalized time array.
    fi : np.ndarray
        Cockx freeze index values.

    References
    ----------
    .. [1] Cockx H, Nonnekes J, Bloem BR, van Wezel R, Cameron I,
       Wang Y. Dealing with the heterogeneous presentations of freezing
       of gait: how reliable are the freezing index and heart rate for
       freezing detection?. Journal of neuroengineering and
       rehabilitation. 2023 Apr 27;20(1):53.
    """
    COCKX_WINDOW_DURATION_S = 3.0
    n = round(COCKX_WINDOW_DURATION_S * fs) + 1
    fi = []
    f, loco_idx, freeze_idx = generate_freqs_locomotion_and_freeze_band_indices(
        n,
        fs,
        loco_f_range=FREQUENCY_RANGE.LOCOMOTOR,
        freeze_f_range=FREQUENCY_RANGE.FREEZING_COCKX,
    )
    w = signal.windows.hann(n, sym=False)
    for center_idx in range(n // 2, len(proxy) - n // 2, n // 2):
        x = proxy[center_idx - n // 2 : center_idx + 2 * (n + 1) // 4]
        if len(x) != n:
            logger.warning(f"Invalid window length detected. Is {len(x)}, expected {n}")
            continue

        X = fft.fft(x * w)
        power_density = np.square(np.abs(X)) / n
        lpd = power_density[loco_idx]
        fpd = power_density[freeze_idx]
        lp = integrate.trapezoid(lpd, f[loco_idx])
        fp = integrate.trapezoid(fpd, f[freeze_idx])
        fi.append((fp / (lp + np.spacing(1))) ** 2)

    t = np.linspace(0, 1, len(fi) + 2)
    return t[1:-1], apply_moore_fi_scaling(np.array(fi))


def compute_zach_fi(
    proxy: np.ndarray, fs: float = 100.0
) -> tuple[np.ndarray, np.ndarray]:
    """Compute FI according to Zach et al.

    This is a Moore FI with window size set to 2 s.

    Parameters
    ----------
    proxy : np.ndarray
        Proxy signal for FI computation.
    fs : float, optional
        Sampling frequency in Hz, by default 100.0.

    Returns
    -------
    t : np.ndarray
        Normalized time array.
    fi : np.ndarray
        Zach freeze index values.
    """
    ZACH_WINDOW_DURATION_S = 2.0
    w = int(fs * ZACH_WINDOW_DURATION_S) + 1
    t, fi = compute_fi_free_window(proxy, w, fs)
    return t, apply_moore_fi_scaling(fi)


def compute_multitaper_fi(
    proxy: np.ndarray,
    fs: float = 100.0,
    dt: float = 5,
    L: int = 4,
    NW: float = 2.5,
    LFTF: float = 3,
    nmaf: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute multitaper FI with L DPSS-tapers.

    Based on the multi-taper spectral estimation algorithm described
    in [1]_.

    Parameters
    ----------
    proxy : np.ndarray
        Proxy signal.
    fs : float, optional
        Sampling frequency in Hz, by default 100.0.
    dt : float, optional
        Time window in seconds, by default 5.
    L : int, optional
        Number of tapers, by default 4.
    NW : float, optional
        DPSS half-bandwidth parameter, by default 2.5.
    LFTF : float, optional
        Locomotion-Freeze-Threshold Frequency in Hz, by default 3.
    nmaf : int, optional
        Moving average filter size. If ``None``, no filtering is applied
        to the FI, by default 5.

    Returns
    -------
    t : np.ndarray
        Normalized time array.
    fi : np.ndarray
        Multitaper freeze index values.

    References
    ----------
    .. [1] Babadi B, Brown EN. A review of multitaper spectral analysis.
       IEEE Transactions on Biomedical Engineering. 2014 Mar
       14;61(5):1555-64.
    """
    F_LOCO = (0.5, LFTF)
    F_FREEZE = (LFTF, 8)

    n = round(dt * fs) + 1
    nfft = 2 ** int(np.log2(n) + 3) - 1
    windows = signal.windows.dpss(n, NW, L, sym=False)

    spectrum = None
    ghost = None
    for win in windows:
        stf = signal.ShortTimeFFT(
            win=win,
            hop=max(n // HOP_DIV, 1),
            fs=fs,
            mfft=nfft,
            scale_to="psd",
            fft_mode="onesided",
        )
        ghost = stf.spectrogram(proxy, detr="linear")

        if spectrum is None:
            spectrum = ghost.copy()
        else:
            spectrum += ghost.copy()

    if ghost is None or spectrum is None:
        raise ValueError("Cannot compute multitaper FI: window list is empty (L=0).")

    locomotor_slc = slice(*[round(nf / (0.5 * fs) * spectrum.shape[0]) for nf in F_LOCO])
    freeze_slc = slice(*[round(nf / (0.5 * fs) * spectrum.shape[0]) for nf in F_FREEZE])
    locomotor_ghost = spectrum[locomotor_slc, :].copy()
    freeze_ghost = spectrum[freeze_slc, :].copy()
    locomotor_f = np.linspace(*F_LOCO, locomotor_ghost.shape[0])
    freeze_f = np.linspace(*F_FREEZE, freeze_ghost.shape[0])
    locomotor_power = integrate.trapezoid(locomotor_ghost, locomotor_f, axis=0)
    freeze_power = integrate.trapezoid(freeze_ghost, freeze_f, axis=0)
    fi = freeze_power / (locomotor_power + np.spacing(1))

    t = stf.t(len(proxy))
    t0 = 0
    t1 = (len(proxy) - 1) / fs
    idx = (t > t0) * (t < t1)

    if isinstance(nmaf, int) and nmaf >= 3:
        mafk = np.ones(nmaf) / nmaf
        fi = np.convolve(fi, mafk, mode="same")

    fi = apply_moore_fi_scaling(fi[idx])
    return np.linspace(0, 1, len(fi)), fi


def generate_freqs_locomotion_and_freeze_band_indices(
    nfft: int,
    fs: float,
    loco_f_range: FREQUENCY_RANGE = FREQUENCY_RANGE.LOCOMOTOR,
    freeze_f_range: FREQUENCY_RANGE = FREQUENCY_RANGE.FREEZING,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate frequencies, locomotion-band indices, and freezing-band indices.

    Parameters
    ----------
    nfft : int
        Number of FFT samples.
    fs : float
        Sampling frequency in Hz.
    loco_f_range : FREQUENCY_RANGE, optional
        Locomotor frequency range, by default FREQUENCY_RANGE.LOCOMOTOR.
    freeze_f_range : FREQUENCY_RANGE, optional
        Freezing frequency range, by default FREQUENCY_RANGE.FREEZING.

    Returns
    -------
    f : np.ndarray
        FFT frequency array.
    loco_idx : np.ndarray
        Boolean mask for the locomotion band.
    freeze_idx : np.ndarray
        Boolean mask for the freezing band.
    """
    f = fft.fftfreq(nfft, d=1.0 / fs)
    loco_idx = (f >= loco_f_range.value[0]) * (f <= loco_f_range.value[1])
    freeze_idx = (f <= freeze_f_range.value[1]) * (f >= freeze_f_range.value[0])
    return f, loco_idx, freeze_idx


def compute_babadi_brown_multitaper_fi(
    proxy: np.ndarray,
    fs: float = 100.0,
    dt: float = 1.0,
    spectral_resolution: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute multitaper FI based on spectral estimation according to Babadi-Brown.

    The number of tapers is derived from the multi-taper spectral
    estimation algorithm described in [1]_.

    Parameters
    ----------
    proxy : np.ndarray
        Proxy signal.
    fs : float, optional
        Sampling frequency in Hz, by default 100.0.
    dt : float, optional
        Time window in seconds, by default 1.0.
    spectral_resolution : float, optional
        Target spectral resolution in Hz, by default 0.05.

    Returns
    -------
    t : np.ndarray
        Normalized time array.
    fi : np.ndarray
        Multitaper freeze index values.

    Raises
    ------
    ValueError
        If the derived alpha parameter is less than one. Increase the
        time window ``dt`` or decrease the target resolution (larger
        value).

    References
    ----------
    .. [1] Babadi B, Brown EN. A review of multitaper spectral analysis.
       IEEE Transactions on Biomedical Engineering. 2014 Mar
       14;61(5):1555-64.
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
) -> tuple[np.ndarray, np.ndarray]:
    """Combine two FI sequences by taking the element-wise maximum.

    It is implicitly assumed that the two FI sequences correspond to the
    same time window and that both feature equispaced samples. The
    sequences are resampled to a common length before the maximum is
    taken at each sample.

    Parameters
    ----------
    lt : list[float]
        Time array for the first (left) FI sequence.
    lfi : list[float]
        First (left) FI array.
    rt : list[float]
        Time array for the second (right) FI sequence.
    rfi : list[float]
        Second (right) FI array.

    Returns
    -------
    t : np.ndarray
        Combined time array.
    fi : np.ndarray
        Element-wise maximum of the two resampled FI sequences.
    """
    N = max(len(lt), len(rt))
    t = np.linspace(min(lt[0], rt[0]), max(lt[-1], rt[-1]), N)
    return t, np.max(np.vstack([np.interp(t, lt, lfi), np.interp(t, rt, rfi)]), axis=0)
