#!/usr/bin/env python3
"""
Plot expected (commanded) points vs detected (vision) in coordinate-grid views.
Reads scan_points_with_grid.csv and writes expected_vs_detected.png.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="scan_points_with_grid.csv")
    ap.add_argument("--out", default="expected_vs_detected.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    required = ["x_mm", "y_mm", "x_m", "y_m"]
    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"Input CSV missing column: {col}")

    expected = np.column_stack([df["x_mm"].values, df["y_mm"].values])
    detected = np.column_stack([df["x_m"].values * 1000.0, df["y_m"].values * 1000.0])

    expected_rel = expected - expected[0]
    detected_rel = detected - detected[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    ax.plot(expected_rel[:, 0], expected_rel[:, 1], "o-", label="Expected (commanded)")
    ax.plot(detected_rel[:, 0], detected_rel[:, 1], "x-", label="Detected (vision)")

    for i in range(len(df)):
        ax.plot(
            [expected_rel[i, 0], detected_rel[i, 0]],
            [expected_rel[i, 1], detected_rel[i, 1]],
            "k-",
            alpha=0.25,
        )

    ax.axis("equal")
    ax.grid(True, alpha=0.35)
    ax.set_xlabel("X (mm, relative)")
    ax.set_ylabel("Y (mm, relative)")
    ax.set_title("Expected vs Detected (Metric XY)")
    ax.legend()

    # Coordinate-grid view in index space
    ax2 = axes[1]
    x_idx = df["x_idx"].to_numpy()
    y_idx = df["y_idx"].to_numpy()
    ax2.scatter(x_idx, y_idx, s=60, marker="s", label="Detected points on commanded grid")

    for i in range(len(df)):
        ax2.text(x_idx[i] + 0.03, y_idx[i] + 0.03, str(int(df["idx"].iloc[i])), fontsize=7, alpha=0.8)

    xmin, xmax = int(np.min(x_idx)), int(np.max(x_idx))
    ymin, ymax = int(np.min(y_idx)), int(np.max(y_idx))
    ax2.set_xticks(np.arange(xmin, xmax + 1, 1))
    ax2.set_yticks(np.arange(ymin, ymax + 1, 1))
    ax2.set_xlim(xmin - 0.5, xmax + 0.5)
    ax2.set_ylim(ymin - 0.5, ymax + 0.5)
    ax2.set_aspect("equal")
    ax2.grid(True, which="major", alpha=0.45)
    ax2.set_xlabel("Grid X index")
    ax2.set_ylabel("Grid Y index")
    ax2.set_title("Coordinate Grid View (x_idx, y_idx)")
    ax2.legend(loc="upper right")

    fig.suptitle("Expected vs Detected Scan Points", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
