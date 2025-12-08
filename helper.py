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


def print_subject_summary(results, subject_name):
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
        print(f"{cond:<12} {r['duration']:.2f}s{'':<4} {r['fs']:.1f}Hz{'':<2} "
              f"{r['rms']:.4f}{'':<4} {r['peak_frequency']:.2f}Hz{'':<6} "
              f"{r['band_power_8_12']:.6f}{'':<6} {r['relative_power_8_12'] * 100:.2f}%")

    print("\n" + "-" * 78)
    print("Comparison: Baseline vs Post-Fatigue")
    print("-" * 78)

    if 'post' in results and 'fat_post' in results:
        baseline_power = results['post']['band_power_8_12']
        fatigue_power = results['fat_post']['band_power_8_12']
        change_pct = ((fatigue_power - baseline_power) / baseline_power) * 100
        print(f"Postural 8-12Hz band power:")
        print(f"  Baseline:     {baseline_power:.6f}")
        print(f"  Post-fatigue: {fatigue_power:.6f}")
        print(f"  Change:       {change_pct:+.2f}%")

    if 'rest' in results and 'fat_rest' in results:
        baseline_power = results['rest']['band_power_8_12']
        fatigue_power = results['fat_rest']['band_power_8_12']
        change_pct = ((fatigue_power - baseline_power) / baseline_power) * 100
        print(f"\nResting 8-12Hz band power:")
        print(f"  Baseline:     {baseline_power:.6f}")
        print(f"  Post-fatigue: {fatigue_power:.6f}")
        print(f"  Change:       {change_pct:+.2f}%")


def print_overall_summary(df):
    print("\n" + "=" * 80)
    print("AGGREGATED DATA SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))


def print_statistical_results(ttest_results, df):
    """Print formatted statistical results."""
    print("\n" + "=" * 80)
    print("PAIRED T-TEST RESULTS: Effect of Forearm Fatigue on Tremor")
    print("=" * 80)

    n_subjects = df['subject'].nunique()
    print(f"\nTotal subjects: {n_subjects}")
    print(f"Subjects: {', '.join(sorted(df['subject'].unique()))}")

    for key, res in ttest_results.items():
        print(f"\n{'-' * 70}")
        print(f"Test: {res['comparison']} ({res['feature']})")
        print(f"{'-' * 70}")
        print(f"  N subjects:      {res['n_subjects']}")
        print(f"  Baseline:        {res['baseline_mean']:.6f} +/- {res['baseline_std']:.6f}")
        print(f"  Post-Fatigue:    {res['fatigue_mean']:.6f} +/- {res['fatigue_std']:.6f}")
        print(f"  Mean difference: {res['mean_diff']:.6f} ({res['percent_change']:+.2f}%)")
        print(f"  t-statistic:     {res['t_statistic']:.4f}")
        print(f"  p-value:         {res['p_value']:.4f}")
        print(f"  Cohen's d:       {res['cohens_d']:.4f}")
        print(f"  Significant:     {'YES' if res['significant'] else 'NO'} (alpha = 0.05)")

    print("\n" + "=" * 80)
    print("HYPOTHESIS EVALUATION")
    print("=" * 80)
    if 'postural_8_12hz' in ttest_results:
        res = ttest_results['postural_8_12hz']
        if res['significant'] and res['mean_diff'] > 0:
            print("H1 SUPPORTED: 8-12 Hz band power significantly higher after fatigue")
        elif res['significant'] and res['mean_diff'] < 0:
            print("H1 REJECTED: 8-12 Hz band power significantly LOWER after fatigue")
        else:
            print("H0 NOT REJECTED: No significant difference in 8-12 Hz band power")
        print(f"  (p = {res['p_value']:.4f}, change = {res['percent_change']:+.2f}%)")


def export_statistics(ttest_results, output_path):
    """Export statistical results to CSV."""
    rows = []
    for key, res in ttest_results.items():
        rows.append(res)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df