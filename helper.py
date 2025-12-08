import pandas as pd
import numpy as np
import sys


def load_accelerometer_data(filepath):
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
    duration = df['time'].iloc[-1] - df['time'].iloc[0]
    n_samples = len(df)
    return n_samples / duration


def print_data_summary(df):
    fs = get_sampling_rate(df)
    duration = df['time'].iloc[-1] - df['time'].iloc[0]

    print(f"Duration: {duration:.2f} s")
    print(f"Samples: {len(df)}")
    print(f"Sampling rate: {fs:.1f} Hz")
    print(f"\nStatistics:")
    print(df[['ax', 'ay', 'az', 'atotal']].describe().round(3))


def extract_signal(df, axis='atotal'):
    return df[axis].values, df['time'].values