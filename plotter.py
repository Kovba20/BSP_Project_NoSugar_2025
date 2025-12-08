from textwrap import shorten

import matplotlib.pyplot as plt


def plot_raw_axes(df):
    """Create a visualization for the raw data for all 3 axes and the total magnitude."""
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
    """Create a visualization for the raw vs processed signal."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(time, raw, color='tab:gray', linewidth=0.5)
    axes[0].set_ylabel('Acceleration (m/s²)')
    axes[0].set_title('Raw signal')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, filtered, color='tab:blue', linewidth=0.5)
    axes[1].set_ylabel('Acceleration (m/s²)')
    axes[1].set_title('Filtered signal')
    axes[1].set_xlabel('Time (s)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_fft(freqs, magnitude, max_freq=30):
    """Create a visualization for fft analysis."""
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
    """Create a visualization the for power spectral density analysis."""
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
    """Create comprehensive visualization for one recording analysis."""
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


def plot_comparison(raw_data, processed_signals, psd_data, results, subject_name):
    """Create comprehensive visualization for subject wide analysis."""

    fig = plt.figure(figsize=(16, 14))

    conditions = ['rest', 'post', 'fat_rest', 'fat_post']
    condition_labels = ['Rest (Baseline)', 'Postural (Baseline)',
                        'Rest (Post-Fatigue)', 'Postural (Post-Fatigue)']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

    # Row 1: Raw signals
    for i, (cond, label, color) in enumerate(zip(conditions, condition_labels, colors)):
        if cond not in raw_data:
            continue
        ax = fig.add_subplot(4, 4, i + 1)
        ax.plot(raw_data[cond]['time'], raw_data[cond]['signal'],
                color=color, linewidth=0.5, alpha=0.8)
        ax.set_title(f'Raw: {label}', fontsize=10)
        ax.set_ylabel('Acc (m/s2)')
        ax.grid(True, alpha=0.3)
        if i >= 2:
            ax.set_xlabel('Time (s)')

    # Row 2: Processed signals
    for i, (cond, label, color) in enumerate(zip(conditions, condition_labels, colors)):
        if cond not in processed_signals:
            continue
        ax = fig.add_subplot(4, 4, i + 5)
        ax.plot(processed_signals[cond]['time'], processed_signals[cond]['signal'],
                color=color, linewidth=0.5)
        ax.set_title(f'Filtered: {label}', fontsize=10)
        ax.set_ylabel('Acc (m/s2)')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time (s)')

    # Row 3: PSD comparison
    ax_psd = fig.add_subplot(4, 2, 5)
    for cond, label, color in zip(conditions, condition_labels, colors):
        if cond not in psd_data:
            continue
        freqs = psd_data[cond]['freqs']
        psd = psd_data[cond]['psd']
        mask = freqs <= 25
        ax_psd.semilogy(freqs[mask], psd[mask], color=color, linewidth=1.5, label=label)
    ax_psd.axvspan(8, 12, alpha=0.15, color='red', label='8-12 Hz band')
    ax_psd.set_xlabel('Frequency (Hz)')
    ax_psd.set_ylabel('PSD (m/s2)^2/Hz')
    ax_psd.set_title('Power Spectral Density Comparison')
    ax_psd.legend(loc='upper right', fontsize=8)
    ax_psd.grid(True, alpha=0.3)

    # Row 3 right: Bar chart of 8-12 Hz band power
    ax_bar = fig.add_subplot(4, 2, 6)
    band_powers = []
    labels_short = []
    bar_colors = []
    for cond, label, color in zip(conditions, ['Rest\nBaseline', 'Post\nBaseline',
                                               'Rest\nFatigue', 'Post\nFatigue'], colors):
        if cond in results:
            band_powers.append(results[cond]['features']['band_power_8_12'])
            labels_short.append(label)
            bar_colors.append(color)

    bars = ax_bar.bar(range(len(band_powers)), band_powers, color=bar_colors, alpha=0.8, edgecolor='black')
    ax_bar.set_xticks(range(len(labels_short)))
    ax_bar.set_xticklabels(labels_short, fontsize=9)
    ax_bar.set_ylabel('8-12 Hz Band Power')
    ax_bar.set_title('Tremor Band Power by Condition')
    ax_bar.grid(True, alpha=0.3, axis='y')

    # Row 4: Feature summary and RMS comparison
    ax_rms = fig.add_subplot(4, 2, 7)
    rms_values = []
    for cond, label, color in zip(conditions, ['Rest\nBaseline', 'Post\nBaseline',
                                               'Rest\nFatigue', 'Post\nFatigue'], colors):
        if cond in results:
            rms_values.append(results[cond]['features']['rms'])

    bars = ax_rms.bar(range(len(rms_values)), rms_values, color=bar_colors, alpha=0.8, edgecolor='black')
    ax_rms.set_xticks(range(len(labels_short)))
    ax_rms.set_xticklabels(labels_short, fontsize=9)
    ax_rms.set_ylabel('RMS Amplitude (m/s2)')
    ax_rms.set_title('RMS Amplitude by Condition')
    ax_rms.grid(True, alpha=0.3, axis='y')

    # Summary text
    ax_text = fig.add_subplot(4, 2, 8)
    ax_text.axis('off')

    summary_lines = [f"Subject: {subject_name}", ""]
    summary_lines.append(f"{'Condition':<18} {'RMS':<10} {'Peak Hz':<10} {'8-12Hz Pwr':<12}")
    summary_lines.append("-" * 50)

    for cond in conditions:
        if cond not in results:
            continue
        f = results[cond]['features']
        summary_lines.append(f"{cond:<18} {f['rms']:.4f}    {f['peak_frequency']:.2f}       {f['band_power_8_12']:.6f}")

    summary_lines.append("")
    summary_lines.append("-" * 50)

    if 'post' in results and 'fat_post' in results:
        bp = results['post']['features']['band_power_8_12']
        fp = results['fat_post']['features']['band_power_8_12']
        change = ((fp - bp) / bp) * 100
        summary_lines.append(f"Postural 8-12Hz change: {change:+.1f}%")

    if 'rest' in results and 'fat_rest' in results:
        bp = results['rest']['features']['band_power_8_12']
        fp = results['fat_rest']['features']['band_power_8_12']
        change = ((fp - bp) / bp) * 100
        summary_lines.append(f"Resting 8-12Hz change:  {change:+.1f}%")

    summary_text = "\n".join(summary_lines)
    ax_text.text(0.1, 0.9, summary_text, transform=ax_text.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.suptitle(f'Finger Tremor Analysis - Subject: {subject_name}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
