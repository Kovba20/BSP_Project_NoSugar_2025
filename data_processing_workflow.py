from helper import load_data
from plotter import visualize_accelerometer


def print_summary(df):
    duration = df['time'].iloc[-1] - df['time'].iloc[0]
    n_samples = len(df)
    fs = n_samples / duration

    print(f"Duration: {duration:.2f} s")
    print(f"Samples: {n_samples}")
    print(f"Sampling rate: {fs:.1f} Hz")
    print(f"\nStatistics:")
    print(df[['ax', 'ay', 'az', 'atotal']].describe().round(3))


if __name__ == "__main__":
    filepath = ".//data//Carolina post .csv"

    df = load_data(filepath)
    print_summary(df)
    visualize_accelerometer(df)