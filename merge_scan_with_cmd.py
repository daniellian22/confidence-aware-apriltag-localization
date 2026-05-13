#!/usr/bin/env python3
"""
Merge scan_points.csv (from extract_scan_points.py) with cmd_log.mat (from MATLAB).
Produces scan_points_with_grid.csv with grid indices and commanded positions.
"""
import argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd_log", default="command_mat_dir/cmd_log_3_20.mat",
                    help="Path to cmd_log.mat from MATLAB run")
    ap.add_argument("--scan_csv", default="scan_points.csv",
                    help="Path to scan_points.csv from extract_scan_points.py")
    ap.add_argument("--out", default="scan_points_with_grid.csv",
                    help="Output CSV path")
    args = ap.parse_args()

    # Load cmd_log.mat: CMD = [pt_idx, x_idx, y_idx, x_mm, y_mm, unix_time]
    mat = loadmat(args.cmd_log)
    CMD = mat["CMD"]
    if CMD.shape[1] != 6:
        raise RuntimeError(f"Expected CMD with 6 columns, got {CMD.shape[1]}")

    # Load scan_points.csv
    scan_df = pd.read_csv(args.scan_csv)
    required = ["idx", "t_sec", "x_m", "y_m", "z_m"]
    for col in required:
        if col not in scan_df.columns:
            raise RuntimeError(f"scan_points.csv missing column: {col}")

    n_cmd = len(CMD)
    n_scan = len(scan_df)
    n = min(n_cmd, n_scan)

    if n_cmd != n_scan:
        print(f"Warning: cmd_log has {n_cmd} points, scan_points has {n_scan}. Using first {n}.")

    # Build merged dataframe
    # CMD columns: pt_idx(0), x_idx(1), y_idx(2), x_mm(3), y_mm(4), unix_time(5)
    merged = pd.DataFrame({
        "idx": np.arange(n),
        "x_idx": CMD[:n, 1].astype(int),
        "y_idx": CMD[:n, 2].astype(int),
        "x_mm": CMD[:n, 3].astype(float),
        "y_mm": CMD[:n, 4].astype(float),
        "x_m": scan_df["x_m"].values[:n],
        "y_m": scan_df["y_m"].values[:n],
        "z_m": scan_df["z_m"].values[:n],
        "t_sec": scan_df["t_sec"].values[:n],
    })

    merged.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(merged)} points")


if __name__ == "__main__":
    main()
