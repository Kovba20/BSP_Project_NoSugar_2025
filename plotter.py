from textwrap import shorten

import matplotlib.pyplot as plt


def plot_raw_axes(df):
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    # Individual axes
    axes[0].plot(df['time'], df['ax'], color='tab:red', linewidth=0.5)
    axes[0].set_ylabel('ax (m/s²)')
    axes[0].set_title('X-axis acceleration')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df['time'], df['ay'], color='tab:green', linewidth=0.5)
    axes[1].set_ylabel('ay (m/s²)')
    axes[1].set_title('Y-axis acceleration')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df['time'], df['az'], color='tab:blue', linewidth=0.5)
    axes[2].set_ylabel('az (m/s²)')
    axes[2].set_title('Z-axis acceleration')
    axes[2].grid(True, alpha=0.3)

    # Total acceleration magnitude
    axes[3].plot(df['time'], df['atotal'], color='tab:purple', linewidth=0.5)
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Total acceleration (m/s²)')
    axes[3].set_title('Acceleration magnitude')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_raw_vs_filtered(time, raw, filtered):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(time, raw, color='tab:gray', linewidth=0.5)
    axes[0].set_ylabel('Acceleration (m/s²)')
    axes[0].set_title('Raw signal')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, filtered, color='tab:blue', linewidth=0.5)
    axes[1].set_ylabel('Acceleration (m/s²)')
    axes[1].set_title('Filtered signal (3-20 Hz bandpass)')
    axes[1].set_xlabel('Time (s)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_fft(freqs, magnitude, max_freq=30):
    fig, ax = plt.subplots(figsize=(10, 5))

    mask = freqs <= max_freq
    ax.plot(freqs[mask], magnitude[mask], color='tab:blue', linewidth=0.8)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.set_title('FFT Spectrum')
    ax.grid(True, alpha=0.3)
    ax.axvspan(8, 12, alpha=0.2, color='red', label='8-12 Hz band')
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_psd(freqs, psd, max_freq=30):
    fig, ax = plt.subplots(figsize=(10, 5))

    mask = freqs <= max_freq
    ax.semilogy(freqs[mask], psd[mask], color='tab:blue', linewidth=0.8)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density (m/s²)²/Hz')
    ax.set_title('Power Spectral Density (Welch)')
    ax.grid(True, alpha=0.3)
    ax.axvspan(8, 12, alpha=0.2, color='red', label='8-12 Hz band')
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_processing_summary(time, raw, filtered, freqs_fft, fft_mag, freqs_psd, psd, features):
    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(time, raw, color='tab:gray', linewidth=0.5)
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.set_title('Raw signal')
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(time, filtered, color='tab:blue', linewidth=0.5)
    ax2.set_ylabel('Acceleration (m/s²)')
    ax2.set_title('Filtered signal (3-20 Hz)')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(3, 2, 3)
    mask_fft = freqs_fft <= 30
    ax3.plot(freqs_fft[mask_fft], fft_mag[mask_fft], color='tab:green', linewidth=0.8)
    ax3.axvspan(8, 12, alpha=0.2, color='red')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Magnitude')
    ax3.set_title('FFT Spectrum')
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(3, 2, 4)
    mask_psd = freqs_psd <= 30
    ax4.semilogy(freqs_psd[mask_psd], psd[mask_psd], color='tab:orange', linewidth=0.8)
    ax4.axvspan(8, 12, alpha=0.2, color='red')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('PSD (m/s²)²/Hz')
    ax4.set_title('Power Spectral Density')
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(3, 2, (5, 6))
    ax5.axis('off')
    feature_text = (
        f"Extracted Features:\n\n"
        f"RMS amplitude:           {features['rms']:.4f} m/s²\n"
        f"Peak frequency:          {features['peak_frequency']:.2f} Hz\n"
        f"8-12 Hz band power:      {features['band_power_8_12']:.6f} (m/s²)²\n"
        f"3-8 Hz band power:       {features['band_power_3_8']:.6f} (m/s²)²\n"
        f"Relative 8-12 Hz power:  {features['relative_power_8_12'] * 100:.2f}%\n"
        f"Total power:             {features['total_power']:.6f} (m/s²)²"
    )
    ax5.text(0.5, 0.5, feature_text, transform=ax5.transAxes, fontsize=12,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    plt.tight_layout()
    plt.show()