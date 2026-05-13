#!/usr/bin/env python3
"""
Verify whether detected scan points match any ordered subset of commanded points.

Method:
1) Load commanded XY points from CMD in .mat (mm)
2) Load detected XY from scan_points.csv (m -> mm)
3) Slide a window of length N_detected over commanded points
4) For each window, fit a 2D similarity transform (scale+rotation+translation)
   using order-preserving correspondences
5) Report the best window and RMSE
"""

import argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat


def fit_similarity_2d(src, dst):
    """Fit dst ~= s * src @ R^T + t, returns (s, R, t, rmse)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    cov = src_c.T @ dst_c / len(src)
    U, S, Vt = np.linalg.svd(cov)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    var_src = np.mean(np.sum(src_c * src_c, axis=1))
    if var_src <= 1e-12:
        raise RuntimeError("Degenerate source points; cannot fit similarity transform")

    scale = np.sum(S) / var_src
    t = dst_mean - scale * (src_mean @ R.T)

    pred = scale * (src @ R.T) + t
    err = np.linalg.norm(pred - dst, axis=1)
    rmse = float(np.sqrt(np.mean(err * err)))
    return scale, R, t, rmse


def evaluate_windows(detected_xy_mm, commanded_xy_mm):
    n_det = len(detected_xy_mm)
    n_cmd = len(commanded_xy_mm)
    if n_det > n_cmd:
        raise RuntimeError(f"Detected points ({n_det}) exceed commanded points ({n_cmd})")

    best = None
    for start in range(0, n_cmd - n_det + 1):
        end = start + n_det
        window = commanded_xy_mm[start:end]

        try:
            scale, R, t, rmse = fit_similarity_2d(detected_xy_mm, window)
        except RuntimeError:
            continue

        if (best is None) or (rmse < best["rmse"]):
            best = {
                "start": start,
                "end": end,
                "rmse": rmse,
                "scale": float(scale),
            }

    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd_log", default="command_mat_dir/cmd_log_3_20.mat")
    ap.add_argument("--scan_csv", default="scan_points.csv")
    args = ap.parse_args()

    cmd_mat = loadmat(args.cmd_log)
    if "CMD" not in cmd_mat:
        raise RuntimeError("CMD matrix not found in command log .mat")
    cmd = cmd_mat["CMD"]
    if cmd.ndim != 2 or cmd.shape[1] < 5:
        raise RuntimeError(f"Unexpected CMD shape: {cmd.shape}")

    commanded_xy_mm = np.column_stack([cmd[:, 3].astype(np.float64), cmd[:, 4].astype(np.float64)])

    scan_df = pd.read_csv(args.scan_csv)
    required = ["x_m", "y_m"]
    for col in required:
        if col not in scan_df.columns:
            raise RuntimeError(f"scan CSV missing column: {col}")

    detected_xy_mm = np.column_stack([
        scan_df["x_m"].to_numpy(dtype=np.float64) * 1000.0,
        scan_df["y_m"].to_numpy(dtype=np.float64) * 1000.0,
    ])

    n_det = len(detected_xy_mm)
    n_cmd = len(commanded_xy_mm)
    print(f"Detected points: {n_det}")
    print(f"Commanded points: {n_cmd}")

    best_fwd = evaluate_windows(detected_xy_mm, commanded_xy_mm)
    best_rev = evaluate_windows(detected_xy_mm[::-1], commanded_xy_mm)

    if best_fwd is None and best_rev is None:
        raise RuntimeError("Could not evaluate any windows")

    if best_rev is None or (best_fwd is not None and best_fwd["rmse"] <= best_rev["rmse"]):
        best = best_fwd
        direction = "forward"
    else:
        best = best_rev
        direction = "reversed"

    print("\nBest ordered match window:")
    print(f"- Direction: {direction}")
    print(f"- Command window (0-based): [{best['start']}, {best['end']})")
    print(f"- Command window (1-based): {best['start'] + 1} .. {best['end']}")
    print(f"- Similarity scale: {best['scale']:.6f}")
    print(f"- XY RMSE after alignment: {best['rmse']:.4f} mm")

    if best["rmse"] < 1.5:
        print("- Verdict: strong geometric consistency")
    elif best["rmse"] < 4.0:
        print("- Verdict: moderate consistency")
    else:
        print("- Verdict: weak consistency")


if __name__ == "__main__":
    main()
