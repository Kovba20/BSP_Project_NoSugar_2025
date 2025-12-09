import os.path
from helper import *
from plotter import *
from processing import *
from statistic_test import perform_paired_ttest


def analyze_recording(filepath, axis='atotal', plot_results=True):
    """Analyze a single accelerometer recording files chosen axis."""
    df = load_accelerometer_data(filepath)
    print_data_summary(df)

    fs = get_sampling_rate(df)
    signal_data, time = extract_signal(df, axis)
    raw = remove_dc_offset(signal_data)

    # Extract features using full pipeline
    features, freqs_fft, fft, freqs_psd, psd, processed, _ = extract_tremor_features(
        signal_data, fs,
        artifact_params={'k': 1.5},
        denoise_params={'window_size': 51, 'threshold_scale': 0.5, 'blend_factor': 0.3}
    )

    # Print extracted features
    print(f"\nExtracted Features ({axis}):")
    print(f"  RMS amplitude:          {features['rms']:.4f} m/s²")
    print(f"  Peak frequency:         {features['peak_frequency']:.2f} Hz")
    print(f"  8-12 Hz band power:     {features['band_power_8_12']:.6f} (m/s²)²")
    print(f"  3-8 Hz band power:      {features['band_power_3_8']:.6f} (m/s²)²")
    print(f"  Relative 8-12 Hz power: {features['relative_power_8_12'] * 100:.2f}%")
    print(f"  Total power:            {features['total_power']:.6f} (m/s²)²")
    if 'artifacts_removed' in features:
        print(f"  Artifacts removed:      {features['artifacts_removed']} ({features['artifacts_percent']:.2f}%)")

    # Generate plots
    if plot_results:
        plot_raw_vs_filtered(time, raw, processed)
        plot_processing_summary(time, raw, processed, freqs_fft, fft, freqs_psd, psd, features)

    return features, processed


def analyze_subject(subject_name, data_dir=".//data//", plot_results=True):
    """Analyze all four conditions for a single subject."""
    conditions = {
        'rest': f'{data_dir}{subject_name} rest.csv',
        'post': f'{data_dir}{subject_name} post.csv',
        'fat_rest': f'{data_dir}{subject_name} fat rest.csv',
        'fat_post': f'{data_dir}{subject_name} fat post.csv'
    }

    results = {}
    raw_data = {}
    processed_signals = {}
    psd_data = {}

    # Process data for each condition
    for condition, filepath in conditions.items():
        try:
            df = load_accelerometer_data(filepath)
            fs = get_sampling_rate(df)
            signal_data, time = extract_signal(df, 'atotal')

            features, freqs_fft, fft_mag, freqs_psd, psd, processed, artifact_mask = extract_tremor_features(
                signal_data, fs,
                artifact_params={'k': 1.5},
                denoise_params={'window_size': 51, 'threshold_scale': 0.5, 'blend_factor': 0.3}
            )

            results[condition] = {
                'subject': subject_name,
                'condition': condition,
                'rms': features['rms'],
                'peak_frequency': features['peak_frequency'],
                'band_power_8_12': features['band_power_8_12'],
                'band_power_3_8': features['band_power_3_8'],
                'relative_power_8_12': features['relative_power_8_12'],
                'total_power': features['total_power'],
                'fs': fs,
                'duration': df['time'].iloc[-1] - df['time'].iloc[0],
                'n_samples': len(df)
            }
            raw_data[condition] = {'time': time, 'signal': signal_data}
            processed_signals[condition] = {'time': time, 'signal': processed}
            psd_data[condition] = {'freqs': freqs_psd, 'psd': psd}
        except Exception as e:
            print(f"Error processing {condition}: {e}")
            continue

    # Show summary and plots
    if plot_results:
        print_subject_summary(results, subject_name)
        plot_subject_analysis_summary(raw_data, processed_signals, psd_data, results, subject_name)

    return results, raw_data, processed_signals, psd_data


def analyze_all_subjects(subjects, data_dir):
    """Aggregate  and analyze data from all subjects."""
    all_results = []

    for subject in subjects:
        results, _, _, _ = analyze_subject(subject, data_dir, False)
        for cond, data in results.items():
            all_results.append(data)

    all_results_df = pd.DataFrame(all_results)
    print_overall_summary(all_results_df)

    # Perform statistical analysis
    ttest_results = perform_paired_ttest(all_results_df)
    print_statistical_results(ttest_results, all_results_df)
    plot_group_results(all_results_df, ttest_results)

    return all_results_df, ttest_results


if __name__ == "__main__":
    # Set file paths and subject list
    data_dir = ".//data//"
    file_name = "Gema fat rest.csv"
    filepath = os.path.join(data_dir, file_name)
    subjects = ['Ari', 'Antonio', 'Candela', 'Carolina', 'Dani', 'Diego', 'Gema', 'Helena', 'Laura', 'Luis', 'María',
                'Miguel', 'Raúl', 'Sánchez', 'Violeta']

    # Analyze single recording
    features, recording_df = analyze_recording(filepath, 'atotal', True)

    # Analyze all conditions for the one subject
    results, raw_data, processed_signals, psd_data = analyze_subject(subjects[0], data_dir, True)

    # Aggregate data and calculate statistics across all subjects
    all_data_df, ttest_results = analyze_all_subjects(subjects, data_dir)
    all_data_df.to_csv('.//results//all_subjects_features.csv', index=False)
    export_statistics(ttest_results, './/results//statistical_results.csv')

