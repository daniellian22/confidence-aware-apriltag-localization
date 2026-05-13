#!/usr/bin/env python3
"""Extract absolute XY coordinates (mm) for a tag from every image in a glob."""
import argparse
import glob
import os
import cv2
import numpy as np
from pupil_apriltags import Detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_glob", required=True)
    ap.add_argument("--intrinsics", default="camera_intrinsics.npz")
    ap.add_argument("--homography", default="pixel_to_grid_H.npz")
    ap.add_argument("--tag_id", type=int, required=True)
    ap.add_argument("--csv_out", default="all_positions.csv")
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
            print(f"  SKIP (unreadable): {os.path.basename(p)}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ds = det.detect(gray, estimate_tag_pose=False)
        t = next((d for d in ds if int(d.tag_id) == args.tag_id), None)
        if t is None:
            print(f"  SKIP (tag {args.tag_id} not found): {os.path.basename(p)}")
            continue

        center = np.array(t.center, dtype=np.float32).reshape(-1, 1, 2)
        und = cv2.undistortPoints(center, K, dist, P=K).reshape(2)
        pt = np.array([[[und[0], und[1]]]], dtype=np.float32)
        xy = cv2.perspectiveTransform(pt, H)[0, 0]
        rows.append((os.path.basename(p), float(xy[0]), float(xy[1])))

    print(f"\n{'image':<22} {'X_mm':>10} {'Y_mm':>10}")
    print("-" * 44)
    for name, x, y in rows:
        print(f"{name:<22} {x:>10.2f} {y:>10.2f}")

    if len(rows) >= 2:
        xs = np.array([r[1] for r in rows])
        ys = np.array([r[2] for r in rows])
        print(f"\nX range: {xs.min():.2f} to {xs.max():.2f} mm  (span {xs.max()-xs.min():.2f} mm)")
        print(f"Y range: {ys.min():.2f} to {ys.max():.2f} mm  (span {ys.max()-ys.min():.2f} mm)")

    with open(args.csv_out, "w") as f:
        f.write("image,X_mm,Y_mm\n")
        for name, x, y in rows:
            f.write(f"{name},{x:.4f},{y:.4f}\n")
    print(f"\nSaved: {args.csv_out}")


if __name__ == "__main__":
    main()
