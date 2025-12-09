import numpy as np
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


def plot_subject_analysis_summary(raw_data, processed_signals, psd_data, results, subject_name):
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
            band_powers.append(results[cond]['band_power_8_12'])
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
            rms_values.append(results[cond]['rms'])

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
        r = results[cond]
        summary_lines.append(f"{cond:<18} {r['rms']:.4f}    {r['peak_frequency']:.2f}       {r['band_power_8_12']:.6f}")

    summary_lines.append("")
    summary_lines.append("-" * 50)

    if 'post' in results and 'fat_post' in results:
        bp = results['post']['band_power_8_12']
        fp = results['fat_post']['band_power_8_12']
        change = ((fp - bp) / bp) * 100
        summary_lines.append(f"Postural 8-12Hz change: {change:+.1f}%")

    if 'rest' in results and 'fat_rest' in results:
        bp = results['rest']['band_power_8_12']
        fp = results['fat_rest']['band_power_8_12']
        change = ((fp - bp) / bp) * 100
        summary_lines.append(f"Resting 8-12Hz change:  {change:+.1f}%")

    summary_text = "\n".join(summary_lines)
    ax_text.text(0.1, 0.9, summary_text, transform=ax_text.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.suptitle(f'Finger Tremor Analysis - Subject: {subject_name}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_group_results(df, ttest_results):
    """Create visualization of group results."""
    fig = plt.figure(figsize=(14, 10))

    subjects = sorted(df['subject'].unique())
    n_subjects = len(subjects)

    # Plot 1: Individual subject 8-12 Hz band power (Postural)
    ax1 = fig.add_subplot(2, 2, 1)
    post_data = df[df['condition'] == 'post'].set_index('subject')['band_power_8_12']
    fat_post_data = df[df['condition'] == 'fat_post'].set_index('subject')['band_power_8_12']

    x = np.arange(n_subjects)
    width = 0.35

    baseline_vals = [post_data.get(s, 0) for s in subjects]
    fatigue_vals = [fat_post_data.get(s, 0) for s in subjects]

    ax1.bar(x - width / 2, baseline_vals, width, label='Baseline', color='#3498db', alpha=0.8)
    ax1.bar(x + width / 2, fatigue_vals, width, label='Post-Fatigue', color='#9b59b6', alpha=0.8)
    ax1.set_ylabel('8-12 Hz Band Power')
    ax1.set_title('Postural: 8-12 Hz Band Power by Subject')
    ax1.set_xticks(x)
    ax1.set_xticklabels(subjects)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Individual subject 8-12 Hz band power (Rest)
    ax2 = fig.add_subplot(2, 2, 2)
    rest_data = df[df['condition'] == 'rest'].set_index('subject')['band_power_8_12']
    fat_rest_data = df[df['condition'] == 'fat_rest'].set_index('subject')['band_power_8_12']

    baseline_vals = [rest_data.get(s, 0) for s in subjects]
    fatigue_vals = [fat_rest_data.get(s, 0) for s in subjects]

    ax2.bar(x - width / 2, baseline_vals, width, label='Baseline', color='#2ecc71', alpha=0.8)
    ax2.bar(x + width / 2, fatigue_vals, width, label='Post-Fatigue', color='#e74c3c', alpha=0.8)
    ax2.set_ylabel('8-12 Hz Band Power')
    ax2.set_title('Rest: 8-12 Hz Band Power by Subject')
    ax2.set_xticks(x)
    ax2.set_xticklabels(subjects)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Paired comparison (Postural) with lines
    ax3 = fig.add_subplot(2, 2, 3)
    for i, subj in enumerate(subjects):
        baseline = post_data.get(subj, np.nan)
        fatigue = fat_post_data.get(subj, np.nan)
        if not np.isnan(baseline) and not np.isnan(fatigue):
            ax3.plot([0, 1], [baseline, fatigue], 'o-', color=f'C{i}',
                     linewidth=2, markersize=8, label=subj)

    ax3.set_xlim(-0.3, 1.3)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['Baseline', 'Post-Fatigue'])
    ax3.set_ylabel('8-12 Hz Band Power')
    ax3.set_title('Postural: Paired Comparison')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    if 'postural_8_12hz' in ttest_results:
        res = ttest_results['postural_8_12hz']
        sig_text = f"p = {res['p_value']:.4f}"
        if res['significant']:
            sig_text += " *"
        ax3.text(0.5, 0.95, sig_text, transform=ax3.transAxes, ha='center',
                 fontsize=11, fontweight='bold')

    # Plot 4: Summary statistics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    summary_lines = ["STATISTICAL SUMMARY", "=" * 40, ""]

    for key, res in ttest_results.items():
        summary_lines.append(f"{res['comparison']}")
        summary_lines.append(f"  Feature: {res['feature']}")
        summary_lines.append(f"  N = {res['n_subjects']}")
        summary_lines.append(f"  Baseline: {res['baseline_mean']:.6f} +/- {res['baseline_std']:.6f}")
        summary_lines.append(f"  Fatigue:  {res['fatigue_mean']:.6f} +/- {res['fatigue_std']:.6f}")
        summary_lines.append(f"  Change:   {res['percent_change']:+.2f}%")
        summary_lines.append(f"  t = {res['t_statistic']:.3f}, p = {res['p_value']:.4f}")
        sig = "SIGNIFICANT" if res['significant'] else "not significant"
        summary_lines.append(f"  Result: {sig}")
        summary_lines.append("")

    summary_text = "\n".join(summary_lines)
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.suptitle(f'Group Analysis: Effect of Forearm Fatigue on Tremor (N={n_subjects})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
