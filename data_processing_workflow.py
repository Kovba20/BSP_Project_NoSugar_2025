from helper import *
from plotter import *
from processing import *


def analyze_recording(filepath, axis='atotal'):
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
    plot_raw_vs_filtered(time, raw, processed)
    plot_processing_summary(time, raw, processed, freqs_fft, fft, freqs_psd, psd, features)

    return features, df


def analyze_subject(subject_name, data_dir="./"):
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
                'features': features,
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

    # Print summary of results
    print_summary(results, subject_name)

    # Generate comparison plot
    plot_comparison(raw_data, processed_signals, psd_data, results, subject_name)

    return results, raw_data, processed_signals, psd_data


if __name__ == "__main__":
    # Set path to a single recording file
    filepath = "data/Gema fat rest.csv"
    # Choose axis to analyze
    axis = "atotal"
    # Analyze single recording
    features, df = analyze_recording(filepath, axis)

    # Set subject name for full analysis
    subject_name = "Laura"
    # Analyze all conditions for the subject
    results, raw_data, processed_signals, psd_data = analyze_subject(subject_name, ".//data//")



