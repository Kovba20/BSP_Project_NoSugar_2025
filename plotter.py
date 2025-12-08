import matplotlib.pyplot as plt


def visualize_accelerometer(df):
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