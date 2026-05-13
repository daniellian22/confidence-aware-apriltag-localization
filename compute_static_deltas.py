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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_glob", required=True)
    ap.add_argument("--intrinsics", default="camera_intrinsics.npz")
    ap.add_argument("--homography", required=True, help="pixel_to_grid_H.npz")
    ap.add_argument("--tag_id", type=int, required=True, help="tag to track")
    args = ap.parse_args()

    imgs = sorted(glob.glob(args.images_glob))
    if not imgs:
        raise RuntimeError(f"No images matched: {args.images_glob}")

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
        X_mm, Y_mm = apply_homography_to_point(und_center[0], und_center[1], H)

        rows.append((os.path.basename(p), X_mm, Y_mm))

    print("grid_xy_mm_per_image")
    for r in rows:
        print(f"{r[0]}: x={r[1]:.2f}, y={r[2]:.2f}")

    print("\nconsecutive_deltas_mm")
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        dx = b[1] - a[1]
        dy = b[2] - a[2]
        dxy = float(np.hypot(dx, dy))
        print(f"{a[0]} -> {b[0]}: dx={dx:.2f}, dy={dy:.2f}, dist_xy={dxy:.2f}")


if __name__ == "__main__":
    main()