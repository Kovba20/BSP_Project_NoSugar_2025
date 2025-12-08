from helper import *
from plotter import *
from processing import *


def analyze_recording(filepath, axis='atotal'):
    df = load_accelerometer_data(filepath)
    print_data_summary(df)

    fs = get_sampling_rate(df)
    signal_data, time = extract_signal(df, axis)

    signal_data = remove_dc_offset(signal_data)

    features, freqs_psd, psd, filtered = extract_tremor_features(signal_data, fs)
    freqs_fft, fft_mag = compute_fft(filtered, fs)

    print(f"\nExtracted Features ({axis}):")
    print(f"  RMS amplitude:          {features['rms']:.4f} m/s²")
    print(f"  Peak frequency:         {features['peak_frequency']:.2f} Hz")
    print(f"  8-12 Hz band power:     {features['band_power_8_12']:.6f} (m/s²)²")
    print(f"  3-8 Hz band power:      {features['band_power_3_8']:.6f} (m/s²)²")
    print(f"  Relative 8-12 Hz power: {features['relative_power_8_12'] * 100:.2f}%")
    print(f"  Total power:            {features['total_power']:.6f} (m/s²)²")

    plot_raw_axes(df)
    plot_raw_vs_filtered(time, signal_data, filtered)
    plot_fft(freqs_fft, fft_mag)
    plot_psd(freqs_psd, psd)
    plot_processing_summary(time, signal_data, filtered, freqs_fft, fft_mag,
                            freqs_psd, psd, features)

    return features, df

if __name__ == "__main__":
    filepath = ".//data//Diego fat post .csv"
    axis = "atotal"

    features, df = analyze_recording(filepath, axis)
