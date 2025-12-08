import pandas as pd
import numpy as np
import sys


def load_accelerometer_data(filepath):
    """Load accelerometer data from CSV file."""
    try:
        df = pd.read_csv(filepath)
        required_cols = ['time', 'ax', 'ay', 'az', 'atotal']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Expected: {required_cols}")
        return df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)


def get_sampling_rate(df):
    """Calculate sampling rate from time column in DataFrame."""
    duration = df['time'].iloc[-1] - df['time'].iloc[0]
    n_samples = len(df)
    return n_samples / duration



def extract_signal(df, axis='atotal'):
    """Extract specified axis signal and time from DataFrame."""
    return df[axis].values, df['time'].values


def print_data_summary(df):
    """Print summary statistics of the accelerometer data from one recording."""
    fs = get_sampling_rate(df)
    duration = df['time'].iloc[-1] - df['time'].iloc[0]

    print(f"Duration: {duration:.2f} s")
    print(f"Samples: {len(df)}")
    print(f"Sampling rate: {fs:.1f} Hz")
    print(f"\nStatistics:")
    print(df[['ax', 'ay', 'az', 'atotal']].describe().round(3))


def print_summary(results, subject_name):
    """Print summary table of extracted features from one subject."""

    print(f"\n{'=' * 70}")
    print(f"SUBJECT: {subject_name}")
    print(f"{'=' * 70}")

    print(
        f"\n{'Condition':<12} {'Duration':<10} {'Fs':<8} {'RMS':<10} {'Peak Freq':<12} {'8-12Hz Power':<14} {'Rel 8-12Hz':<12}")
    print("-" * 78)

    for cond in ['rest', 'post', 'fat_rest', 'fat_post']:
        if cond not in results:
            continue
        r = results[cond]
        f = r['features']
        print(f"{cond:<12} {r['duration']:.2f}s{'':<4} {r['fs']:.1f}Hz{'':<2} "
              f"{f['rms']:.4f}{'':<4} {f['peak_frequency']:.2f}Hz{'':<6} "
              f"{f['band_power_8_12']:.6f}{'':<6} {f['relative_power_8_12'] * 100:.2f}%")

    print("\n" + "-" * 78)
    print("Comparison: Baseline vs Post-Fatigue")
    print("-" * 78)

    if 'post' in results and 'fat_post' in results:
        baseline_power = results['post']['features']['band_power_8_12']
        fatigue_power = results['fat_post']['features']['band_power_8_12']
        change_pct = ((fatigue_power - baseline_power) / baseline_power) * 100
        print(f"Postural 8-12Hz band power:")
        print(f"  Baseline:     {baseline_power:.6f}")
        print(f"  Post-fatigue: {fatigue_power:.6f}")
        print(f"  Change:       {change_pct:+.2f}%")

    if 'rest' in results and 'fat_rest' in results:
        baseline_power = results['rest']['features']['band_power_8_12']
        fatigue_power = results['fat_rest']['features']['band_power_8_12']
        change_pct = ((fatigue_power - baseline_power) / baseline_power) * 100
        print(f"\nResting 8-12Hz band power:")
        print(f"  Baseline:     {baseline_power:.6f}")
        print(f"  Post-fatigue: {fatigue_power:.6f}")
        print(f"  Change:       {change_pct:+.2f}%")

