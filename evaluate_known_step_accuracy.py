#!/usr/bin/env python3
import argparse
import glob
import os
import cv2
import numpy as np
from pupil_apriltags import Detector


def undistort_points(pts_uv, K, dist):
    pts = np.asarray(pts_uv, dtype=np.float32).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist, P=K)
    return und.reshape(-1, 2)


def apply_homography_to_point(u, v, H):
    p = np.array([[[u, v]]], dtype=np.float32)
    out = cv2.perspectiveTransform(p, H)[0, 0]
    return float(out[0]), float(out[1])


def choose_axis(xs, ys, axis):
    if axis in ("x", "y"):
        return axis
    span_x = float(np.max(xs) - np.min(xs))
    span_y = float(np.max(ys) - np.min(ys))
    return "x" if span_x >= span_y else "y"


def summarize(name, arr):
    if len(arr) == 0:
        return
    arr = np.asarray(arr, dtype=np.float64)
    print(
        f"{name}: mean={np.mean(arr):.3f}, median={np.median(arr):.3f}, "
        f"std={np.std(arr):.3f}, min={np.min(arr):.3f}, max={np.max(arr):.3f}"
    )


def main():
    ap = argparse.ArgumentParser(description="Evaluate known-step XY accuracy from image sequence.")
    ap.add_argument("--images_glob", required=True, help="Ordered image sequence glob.")
    ap.add_argument("--intrinsics", default="camera_intrinsics.npz")
    ap.add_argument("--homography", required=True, help="pixel_to_grid_H.npz")
    ap.add_argument("--tag_id", type=int, required=True, help="tag to track")
    ap.add_argument(
        "--axis",
        choices=["x", "y", "auto"],
        default="auto",
        help="Expected motion axis in grid frame.",
    )
    ap.add_argument(
        "--expected_step_mm",
        type=float,
        required=True,
        help="Expected per-step distance in mm. Use signed value if direction is known.",
    )
    ap.add_argument(
        "--csv_out",
        default="",
        help="Optional per-step CSV output path.",
    )
    args = ap.parse_args()

    imgs = sorted(glob.glob(args.images_glob))
    if len(imgs) < 2:
        raise RuntimeError(
            f"Need at least 2 images for step evaluation; matched {len(imgs)} with --images_glob '{args.images_glob}'"
        )

    intr = np.load(args.intrinsics)
    K = intr["K"].astype(np.float64)
    dist = intr["dist"].astype(np.float64)

    Hdata = np.load(args.homography)
    H = Hdata["H"].astype(np.float64)

    det = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    rows = []
    for p in imgs:
        img = cv2.imread(p)
        if img is None:
            print(f"{os.path.basename(p)}: unreadable")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ds = det.detect(gray, estimate_tag_pose=False)

        target = None
        for d in ds:
            if int(d.tag_id) == args.tag_id:
                target = d
                break

        if target is None:
            print(f"{os.path.basename(p)}: missing tag {args.tag_id}")
            continue

        center = np.array(target.center, dtype=np.float64).reshape(1, 2)
        und_center = undistort_points(center, K, dist)[0]
        x_mm, y_mm = apply_homography_to_point(und_center[0], und_center[1], H)
        rows.append((os.path.basename(p), x_mm, y_mm))

    if len(rows) < 2:
        raise RuntimeError("Insufficient detections after filtering images")

    xs = np.array([r[1] for r in rows], dtype=np.float64)
    ys = np.array([r[2] for r in rows], dtype=np.float64)
    axis = choose_axis(xs, ys, args.axis)
    primary = xs if axis == "x" else ys
    secondary = ys if axis == "x" else xs

    total_motion = float(primary[-1] - primary[0])
    direction = 1.0 if total_motion >= 0 else -1.0
    expected = float(args.expected_step_mm)
    if expected > 0:
        expected *= direction

    print(f"axis_used={axis}")
    print(f"expected_step_mm={expected:.3f}")
    print(f"samples_used={len(rows)}")

    header = (
        "step,from,to,dx_primary_mm,dx_secondary_mm,"
        "expected_primary_mm,error_mm,abs_error_mm"
    )
    print("\n" + header)

    lines = [header]
    step_errors = []
    abs_step_errors = []
    orth_drift = []

    for i in range(len(rows) - 1):
        a = rows[i]
        b = rows[i + 1]
        dp = float(primary[i + 1] - primary[i])
        ds = float(secondary[i + 1] - secondary[i])
        err = dp - expected
        aerr = abs(err)

        step_errors.append(err)
        abs_step_errors.append(aerr)
        orth_drift.append(abs(ds))

        line = (
            f"{i + 1},{a[0]},{b[0]},{dp:.3f},{ds:.3f},"
            f"{expected:.3f},{err:.3f},{aerr:.3f}"
        )
        lines.append(line)
        print(line)

    print("\nsummary")
    summarize("step_error_mm", np.array(step_errors, dtype=np.float64))
    summarize("abs_step_error_mm", np.array(abs_step_errors, dtype=np.float64))
    summarize("orthogonal_drift_abs_mm", np.array(orth_drift, dtype=np.float64))

    if args.csv_out:
        with open(args.csv_out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nwrote_csv={args.csv_out}")


if __name__ == "__main__":
    main()
