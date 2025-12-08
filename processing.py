import numpy as np
from scipy import signal, stats, ndimage
from scipy.fft import fft, fftfreq


def iqr_artifact_removal(data, k=1.5):
    """
    Remove artifacts using Interquartile Range method.

    Points outside [Q1 - k*IQR, Q3 + k*IQR] are considered artifacts
    and replaced via linear interpolation.

    Parameters
    ----------
    data : array
        Input signal
    k : float
        IQR multiplier (1.5 = standard outlier, 3.0 = extreme outlier)

    Returns
    -------
    cleaned : array
        Signal with artifacts replaced
    artifact_mask : array
        Boolean mask indicating artifact locations
    """
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    artifact_mask = (data < lower_bound) | (data > upper_bound)

    cleaned = data.copy()
    if artifact_mask.any():
        valid_indices = np.where(~artifact_mask)[0]
        artifact_indices = np.where(artifact_mask)[0]
        cleaned[artifact_mask] = np.interp(artifact_indices, valid_indices, data[valid_indices])

    return cleaned, artifact_mask


def adaptive_local_denoise(data, window_size=51, threshold_scale=0.5, blend_factor=0.3):
    """
    Adaptive local denoising that preserves high-activity regions.

    Smooths only where local variance is low (noise), preserves where
    local variance is high (actual signal like tremor).

    Parameters
    ----------
    data : array
        Input signal
    window_size : int
        Size of local window for variance estimation (must be odd)
    threshold_scale : float
        Scaling factor for adaptive threshold (lower = more aggressive)
    blend_factor : float
        How much smoothing to apply where noise detected (0-1)

    Returns
    -------
    denoised : array
        Denoised signal
    """
    if window_size % 2 == 0:
        window_size += 1

    # Compute local statistics using scipy.ndimage for efficiency
    local_mean = ndimage.uniform_filter1d(data, size=window_size, mode='reflect')
    local_sq_mean = ndimage.uniform_filter1d(data ** 2, size=window_size, mode='reflect')
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))

    global_std = np.std(data)

    # Adaptive threshold: higher in active regions, lower in quiet regions
    threshold = threshold_scale * (local_std + 0.5 * global_std)

    # Extract high-frequency component
    smoothed = ndimage.uniform_filter1d(data, size=5, mode='reflect')
    high_freq = data - smoothed

    # Apply selective smoothing
    denoised = data.copy()
    noise_mask = np.abs(high_freq) < threshold
    denoised[noise_mask] = (1 - blend_factor) * data[noise_mask] + blend_factor * smoothed[noise_mask]

    return denoised


def bandpass_filter(data, fs, lowcut=3.0, highcut=20.0, order=4):
    """
    Zero-phase Butterworth bandpass filter.
    """
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)


def remove_dc_offset(data):
    """Remove mean (DC component) from signal."""
    return data - np.mean(data)


def normalize_zscore(data):
    """Normalize signal to zero mean and unit variance."""
    return (data - np.mean(data)) / np.std(data)


def compute_fft(data, fs):
    """Compute FFT magnitude spectrum."""
    n = len(data)
    freqs = fftfreq(n, 1 / fs)[:n // 2]
    fft_values = fft(data)
    magnitude = 2.0 / n * np.abs(fft_values[:n // 2])
    return freqs, magnitude


def compute_psd_welch(data, fs, nperseg=256):
    """Compute Power Spectral Density using Welch's method."""
    freqs, psd = signal.welch(data, fs, nperseg=nperseg)
    return freqs, psd


def compute_band_power(freqs, psd, low_freq, high_freq):
    """Compute power in a specific frequency band."""
    idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
    return np.trapezoid(psd[idx_band], freqs[idx_band])


def compute_total_power(freqs, psd):
    """Compute total power across all frequencies."""
    return np.trapezoid(psd, freqs)


def compute_relative_band_power(freqs, psd, low_freq, high_freq):
    """Compute band power relative to total power."""
    band_power = compute_band_power(freqs, psd, low_freq, high_freq)
    total_power = compute_total_power(freqs, psd)
    return band_power / total_power if total_power > 0 else 0


def find_peak_frequency(freqs, psd, low_freq=3.0, high_freq=20.0):
    """Find dominant frequency within a band."""
    idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
    freqs_band = freqs[idx_band]
    psd_band = psd[idx_band]
    peak_idx = np.argmax(psd_band)
    return freqs_band[peak_idx]


def compute_rms(data):
    """Compute Root Mean Square amplitude."""
    return np.sqrt(np.mean(data ** 2))


def preprocess_signal(data, fs, artifact_params=None, denoise_params=None):
    """
    Complete preprocessing pipeline: artifact removal -> denoising -> filtering.

    Parameters
    ----------
    data : array
        Raw input signal
    fs : float
        Sampling frequency
    artifact_params : dict
        Parameters for artifact removal
    denoise_params : dict
        Parameters for denoising

    Returns
    -------
    processed : array
        Preprocessed signal
    artifact_mask : array
        Boolean mask of detected artifacts (or None)
    """
    artifact_params = artifact_params or {}
    denoise_params = denoise_params or {}

    processed = remove_dc_offset(data)
    artifact_mask = None

    # Step 1: Artifact removal
    processed, artifact_mask = iqr_artifact_removal(processed, **artifact_params)

    # Step 2: Denoising
    processed = adaptive_local_denoise(processed, **denoise_params)

    # Step 3: Bandpass filter
    processed = bandpass_filter(processed, fs)

    return processed, artifact_mask


def extract_tremor_features(data, fs, artifact_params=None, denoise_params=None):
    """
    Extract tremor features from signal with full preprocessing.

    Returns
    -------
    features : dict
        Extracted tremor features
    freqs : array
        Frequency axis for PSD
    psd : array
        Power spectral density
    processed : array
        Preprocessed signal
    artifact_mask : array
        Detected artifact locations
    """
    processed, artifact_mask = preprocess_signal(data, fs, artifact_params, denoise_params)

    freqs_fft, fft = compute_fft(processed, fs)
    freqs_psd, psd = compute_psd_welch(processed, fs)

    features = {
        'rms': compute_rms(processed),
        'peak_frequency': find_peak_frequency(freqs_psd, psd),
        'band_power_8_12': compute_band_power(freqs_psd, psd, 8, 12),
        'band_power_3_8': compute_band_power(freqs_psd, psd, 3, 8),
        'relative_power_8_12': compute_relative_band_power(freqs_psd, psd, 8, 12),
        'total_power': compute_total_power(freqs_psd, psd)
    }

    if artifact_mask is not None:
        features['artifacts_removed'] = artifact_mask.sum()
        features['artifacts_percent'] = 100 * artifact_mask.sum() / len(data)

    return features, freqs_fft, fft, freqs_psd, psd, processed, artifact_mask