import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import serial
from numpy.polynomial import Polynomial


def main():
    func = sys.argv[1]

    if func == "collect":
        collect_data(sys.argv[2] if len(sys.argv) > 2 else "accel")
    elif func == "display":
        if len(sys.argv) < 4:
            print("Usage: python main.py display <axis> <csv_path>")
            print("Example: python main.py display x data-1234567890.csv")
            return
        axis = sys.argv[2]
        csv_path = sys.argv[3]
        if axis == "3d":
            displayData3d(csv_path)
        else:
            displayData(axis, csv_path)
    elif func == "fit":
        fit_data(sys.argv[2], sys.argv[3])
    else:
        print(f"Unknown function: {func}")
        print("Usage: python main.py [collect|display|fit]")
        print("Example: python main.py collect [accel|gyro]")
        print("Example: python main.py display [x|y|z|all|3d] data-1234567890.csv")
        print("Example: python main.pay fit [x|y|z] data-1234567890.csv")


def displayData3d(csv_path: str):
    # 3D vector plot: each sample is a vector from origin to (x,y,z)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    required = ["accelX", "accelY", "accelZ"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing}. Available: {list(df.columns)}")

    x = df["accelX"].astype(float).to_numpy()
    y = df["accelY"].astype(float).to_numpy()
    z = df["accelZ"].astype(float).to_numpy()

    # Downsample if very large so the plot stays responsive
    n = len(df)
    step = max(1, n // 2000)
    x, y, z = x[::step], y[::step], z[::step]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Draw vectors from origin using quiver
    zeros = np.zeros_like(x)
    ax.quiver(
        zeros,
        zeros,
        zeros,
        x,
        y,
        z,
        length=1,
        normalize=False,
        linewidth=0.5,
    )

    # Also plot the tip trajectory for context
    ax.plot(x, y, z, label="Tip trajectory", linewidth=1.0)

    ax.set_title("Accel vectors (downsampled)" if step > 1 else "Accel vectors")
    ax.set_xlabel("accelX")
    ax.set_ylabel("accelY")
    ax.set_zlabel("accelZ")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return


def displayData(axis: str, csv_path: str) -> None:
    """Display plots for accel data from a CSV file.

    axis:
      - one of {"x", "y", "z"} (case-insensitive) for a 2D timeseries plot
      - "all" to show all graphs in one window

    csv_path: path to a CSV with header: accelX, accelY, accelZ
    """
    axis = axis.strip().lower()

    if axis == "all":
        xVals = []
        yVals = []
        zVals = []

        with open(csv_path, "r") as f:
            next(f, None)
            for line in f:
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                try:
                    xVals.append(float(parts[0]))
                    yVals.append(float(parts[1]))
                    zVals.append(float(parts[2]))
                except ValueError:
                    continue

        xSeries = list(range(len(xVals)))

        # Draw one figure per axis (3 total)
        for label, arr in (("X", xVals), ("Y", yVals), ("Z", zVals)):
            plt.figure()
            plt.plot(xSeries, arr)
            plt.title(f"Accel {label} vs Sample")
            plt.xlabel("xSeries")
            plt.ylabel(f"accel{label}")
            plt.grid(True)
            plt.tight_layout()

        plt.show()
        return

    axis_to_col = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_to_col:
        raise ValueError("axis must be one of 'x', 'y', 'z', or 'all'")

    ys = []
    with open(csv_path, "r") as f:
        # skip header
        next(f, None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                ys.append(float(parts[axis_to_col[axis]]))
            except ValueError:
                # ignore non-numeric rows
                continue

    xs = list(range(len(ys)))

    plt.figure()
    plt.plot(xs, ys)
    plt.title(f"Accel {axis.upper()} vs Sample")
    plt.xlabel("Sample")
    plt.ylabel(f"Accel{axis.upper()}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def collect_data(data: str):
    port = "/dev/ttyACM1"
    baud = 115200

    ser = serial.Serial(port, baud)
    ser.flush()

    time.sleep(2)

    print(f"Connected to {port} at {baud} baud rate...")

    with open(f"data-{data}-{time.time()}.csv", "w") as file:
        file.write("dataType,accelX,accelY,accelZ\n")

        dataType = 0
        if data == "gyro":
            dataType = 1

        try:
            while True:
                if ser.in_waiting > 0:
                    line = ser.readline().decode("utf-8").rstrip()

                    if line.startswith(str(dataType)):
                        file.write(line + "\n")

        except KeyboardInterrupt:
            print("Exiting...")
            ser.close()
            file.close()


def fit_data(data_point, csv_path):
    data = pd.read_csv(csv_path)
    print(data.head())

    if data_point not in data.columns:
        raise KeyError(
            f"Column '{data_point}' not found. Available columns: {list(data.columns)}"
        )

    data_val = data[data_point].values

    x = np.arange(len(data_val), dtype=float)
    p_fit = Polynomial.fit(x, data_val.astype(float), deg=3)

    plt.figure()
    plt.plot(x, data_val.astype(float), label="Data", marker="o", linestyle="")
    plt.plot(x, p_fit(x), label="Polynomial Fit", color="red")
    plt.title(f"Polynomial Fit for {data_point}")
    plt.xlabel("Sample")
    plt.ylabel(data_point)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")
        sys.exit(0)
