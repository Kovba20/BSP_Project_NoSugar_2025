import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq


def bandpass_filter(data, fs, lowcut=3.0, highcut=20.0, order=4):
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)


def remove_dc_offset(data):
    return data - np.mean(data)


def normalize_zscore(data):
    return (data - np.mean(data)) / np.std(data)


def compute_fft(data, fs):
    n = len(data)
    freqs = fftfreq(n, 1/fs)[:n//2]
    fft_values = fft(data)
    magnitude = 2.0/n * np.abs(fft_values[:n//2])
    return freqs, magnitude


def compute_psd_welch(data, fs, nperseg=256):
    freqs, psd = signal.welch(data, fs, nperseg=nperseg)
    return freqs, psd


def compute_band_power(freqs, psd, low_freq, high_freq):
    idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
    band_power = np.trapz(psd[idx_band], freqs[idx_band])
    return band_power


def compute_total_power(freqs, psd):
    return np.trapz(psd, freqs)


def compute_relative_band_power(freqs, psd, low_freq, high_freq):
    band_power = compute_band_power(freqs, psd, low_freq, high_freq)
    total_power = compute_total_power(freqs, psd)
    return band_power / total_power if total_power > 0 else 0


def find_peak_frequency(freqs, psd, low_freq=3.0, high_freq=20.0):
    idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
    freqs_band = freqs[idx_band]
    psd_band = psd[idx_band]
    peak_idx = np.argmax(psd_band)
    return freqs_band[peak_idx]


def compute_rms(data):
    return np.sqrt(np.mean(data**2))


def extract_tremor_features(data, fs):
    filtered = bandpass_filter(data, fs)
    freqs, psd = compute_psd_welch(filtered, fs)
    
    features = {
        'rms': compute_rms(filtered),
        'peak_frequency': find_peak_frequency(freqs, psd),
        'band_power_8_12': compute_band_power(freqs, psd, 8, 12),
        'band_power_3_8': compute_band_power(freqs, psd, 3, 8),
        'relative_power_8_12': compute_relative_band_power(freqs, psd, 8, 12),
        'total_power': compute_total_power(freqs, psd)
    }
    return features, freqs, psd, filtered
