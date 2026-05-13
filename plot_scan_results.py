#!/usr/bin/env python3
"""
Plot scan point results: positions, distances between points, and error vs expected spacing.
Expected: Y step = 1 mm, X step = 7.6 mm.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

EXPECTED_Y_MM = 1.0
EXPECTED_X_MM = 7.6


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Convert to mm for distance comparison
    df["x_mm_meas"] = df["x_m"] * 1000
    df["y_mm_meas"] = df["y_m"] * 1000
    df["z_mm_meas"] = df["z_m"] * 1000
    return df


def compute_distances(df):
    """Compute distances between consecutive points and classify as X-step or Y-step."""
    pos = df[["x_mm_meas", "y_mm_meas", "z_mm_meas"]].values
    x_idx = df["x_idx"].values
    y_idx = df["y_idx"].values

    dists = []
    dx_arr = []
    dy_arr = []
    dz_arr = []
    step_type = []  # "x" or "y"
    pair_idx = []

    for i in range(len(df) - 1):
        d = np.linalg.norm(pos[i + 1] - pos[i])
        dx = pos[i + 1, 0] - pos[i, 0]
        dy = pos[i + 1, 1] - pos[i, 1]
        dz = pos[i + 1, 2] - pos[i, 2]

        dists.append(d)
        dx_arr.append(dx)
        dy_arr.append(dy)
        dz_arr.append(dz)
        pair_idx.append(i)

        # X-step: x_idx changed
        if x_idx[i + 1] != x_idx[i]:
            step_type.append("x")
        else:
            step_type.append("y")

    return (
        np.array(dists),
        np.array(dx_arr),
        np.array(dy_arr),
        np.array(dz_arr),
        step_type,
        np.array(pair_idx),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="scan_points_with_grid.csv")
    ap.add_argument("--out", default="scan_analysis.png")
    args = ap.parse_args()

    df = load_data(args.csv)
    n = len(df)

    dists, dx_arr, dy_arr, dz_arr, step_type, pair_idx = compute_distances(df)
    y_mask = np.array([t == "y" for t in step_type])
    x_mask = np.array([t == "x" for t in step_type])

    y_dists = dists[y_mask]
    x_dists = dists[x_mask]
    y_errors = y_dists - EXPECTED_Y_MM
    x_errors = x_dists - EXPECTED_X_MM

    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # --- 1. 3D scatter of positions ---
    ax1 = fig.add_subplot(gs[0, :], projection="3d")
    ax1.scatter(df["x_mm_meas"], df["y_mm_meas"], df["z_mm_meas"], c=df["idx"], cmap="viridis", s=40)
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel("Z (mm)")
    ax1.set_title("3D Scan Point Positions (measured, origin frame)")

    # --- 2. XY projection ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(df["x_mm_meas"], df["y_mm_meas"], c=df["idx"], cmap="viridis", s=40)
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.set_title("XY Projection")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    # --- 3. XZ projection ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(df["x_mm_meas"], df["z_mm_meas"], c=df["idx"], cmap="viridis", s=40)
    ax3.set_xlabel("X (mm)")
    ax3.set_ylabel("Z (mm)")
    ax3.set_title("XZ Projection")
    ax3.grid(True, alpha=0.3)

    # --- 4. YZ projection ---
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.scatter(df["y_mm_meas"], df["z_mm_meas"], c=df["idx"], cmap="viridis", s=40)
    ax4.set_xlabel("Y (mm)")
    ax4.set_ylabel("Z (mm)")
    ax4.set_title("YZ Projection")
    ax4.grid(True, alpha=0.3)

    # --- 5. Positions vs point index (along axes) ---
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(df["idx"], df["x_mm_meas"], "o-", label="X", markersize=4)
    ax5.plot(df["idx"], df["y_mm_meas"], "s-", label="Y", markersize=4)
    ax5.plot(df["idx"], df["z_mm_meas"], "^-", label="Z", markersize=4)
    ax5.set_xlabel("Point index")
    ax5.set_ylabel("Position (mm)")
    ax5.set_title("Position vs Scan Order")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # --- 6. Distance between consecutive points ---
    ax6 = fig.add_subplot(gs[2, 1])
    colors = ["#2ecc71" if t == "y" else "#e74c3c" for t in step_type]
    bars = ax6.bar(pair_idx, dists, color=colors, alpha=0.8)
    ax6.axhline(EXPECTED_Y_MM, color="green", linestyle="--", alpha=0.7, label=f"Expected Y: {EXPECTED_Y_MM} mm")
    ax6.axhline(EXPECTED_X_MM, color="red", linestyle="--", alpha=0.7, label=f"Expected X: {EXPECTED_X_MM} mm")
    ax6.set_xlabel("Consecutive pair index")
    ax6.set_ylabel("Distance (mm)")
    ax6.set_title("Distance Between Consecutive Points\n(green=Y-step, red=X-step)")
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis="y")

    # --- 7. Error from expected (Y and X steps) ---
    ax7 = fig.add_subplot(gs[2, 2])
    if len(y_dists) > 0:
        ax7.bar(np.arange(len(y_dists)), y_errors, color="green", alpha=0.6, label=f"Y-step (n={len(y_dists)})")
    if len(x_dists) > 0:
        offset = len(y_dists)
        ax7.bar(offset + np.arange(len(x_dists)), x_errors, color="red", alpha=0.6, label=f"X-step (n={len(x_dists)})")
    ax7.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax7.set_xlabel("Step index")
    ax7.set_ylabel("Error (mm)\n(measured - expected)")
    ax7.set_title("Error from Expected Spacing")
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Scan Analysis: {n} points | Expected Y={EXPECTED_Y_MM} mm, X={EXPECTED_X_MM} mm", fontsize=12)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}")

    # --- Second figure: per-axis deltas and error histograms ---
    fig2, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    if len(y_dists) > 0:
        y_dx = np.array(dx_arr)[y_mask]
        y_dy = np.array(dy_arr)[y_mask]
        y_dz = np.array(dz_arr)[y_mask]
        x_pos = np.arange(len(y_dists))
        w = 0.25
        ax.bar(x_pos - w, y_dx, w, label="dX", alpha=0.8)
        ax.bar(x_pos, y_dy, w, label="dY", alpha=0.8)
        ax.bar(x_pos + w, y_dz, w, label="dZ", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Y-step index")
    ax.set_ylabel("Delta (mm)")
    ax.set_title("Per-axis change for Y-steps (expected 1 mm total)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[0, 1]
    if len(y_dists) > 0:
        ax.hist(y_errors, bins=min(15, max(5, len(y_dists) // 2)), color="green", alpha=0.6, edgecolor="black")
    ax.axvline(0, color="black", linestyle="--")
    ax.set_xlabel("Error (mm)")
    ax.set_ylabel("Count")
    ax.set_title(f"Y-step error histogram (expected {EXPECTED_Y_MM} mm)")

    ax = axes[1, 0]
    if len(x_dists) > 0:
        x_dx = np.array(dx_arr)[x_mask]
        x_dy = np.array(dy_arr)[x_mask]
        x_dz = np.array(dz_arr)[x_mask]
        ax.bar([0], [x_dx[0]], 0.25, label="dX", alpha=0.8)
        ax.bar([0.25], [x_dy[0]], 0.25, label="dY", alpha=0.8)
        ax.bar([0.5], [x_dz[0]], 0.25, label="dZ", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(EXPECTED_X_MM, color="red", linestyle="--", alpha=0.7, label=f"Expected {EXPECTED_X_MM} mm")
    ax.set_xlabel("X-step")
    ax.set_ylabel("Delta (mm)")
    ax.set_title("Per-axis change for X-step (expected 7.6 mm total)")
    ax.legend()
    ax.set_xticks([])

    ax = axes[1, 1]
    if len(x_dists) > 0:
        ax.bar([0], x_errors, color="red", alpha=0.6, edgecolor="black")
        ax.axvline(0, color="black", linestyle="--")
    ax.set_xlabel("X-step error")
    ax.set_ylabel("Error (mm)")
    ax.set_title(f"X-step error (expected {EXPECTED_X_MM} mm)")
    ax.set_xticks([])

    fig2.suptitle("Distance Components and Error Distribution", fontsize=12)
    fig2.tight_layout()
    out2 = args.out.replace(".png", "_detail.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")

    # --- Summary statistics ---
    print("\n--- Summary ---")
    if len(y_dists) > 0:
        print(f"Y-step (expected {EXPECTED_Y_MM} mm): mean={y_dists.mean():.3f} mm, std={y_dists.std():.3f} mm, "
              f"error mean={y_errors.mean():.3f} mm, error std={y_errors.std():.3f} mm")
    if len(x_dists) > 0:
        print(f"X-step (expected {EXPECTED_X_MM} mm): mean={x_dists.mean():.3f} mm, std={x_dists.std():.3f} mm, "
              f"error mean={x_errors.mean():.3f} mm, error std={x_errors.std():.3f} mm")


if __name__ == "__main__":
    main()
