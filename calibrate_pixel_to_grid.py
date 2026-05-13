#!/usr/bin/env python3
import argparse
import csv
import os
import cv2
import numpy as np
from pupil_apriltags import Detector


def undistort_points(pts_uv, K, dist):
    pts = np.asarray(pts_uv, dtype=np.float32).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist, P=K)
    return und.reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--tag_family", default="tag36h11")
    ap.add_argument("--tag_id", default="auto")
    ap.add_argument("--out_h", default="pixel_to_grid_H.npz")
    ap.add_argument("--out_csv", default="pairs_with_pixels.csv")
    args = ap.parse_args()

    intr = np.load(args.intrinsics)
    K = intr["K"]
    dist = intr["dist"]

    det = Detector(families=args.tag_family)

    pixel_pts = []
    world_pts = []

    with open(args.pairs_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.join(args.images_dir, row["image"])
            img = cv2.imread(img_path)

            if img is None:
                print(f"Skipping {img_path} (not found)")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detections = det.detect(gray, estimate_tag_pose=False)

            chosen = None

            if args.tag_id == "auto":
                if len(detections) > 0:
                    chosen = detections[0]
            else:
                for d in detections:
                    if int(d.tag_id) == int(args.tag_id):
                        chosen = d
                        break

            if chosen is None:
                print(f"No tag found in {img_path}")
                continue

            u, v = chosen.center
            und = undistort_points([[u, v]], K, dist)[0]

            pixel_pts.append([und[0], und[1]])
            world_pts.append([float(row["x_mm"]), float(row["y_mm"])])

    pixel_pts = np.array(pixel_pts)
    world_pts = np.array(world_pts)

    if len(pixel_pts) < 4:
        raise RuntimeError("Need at least 4 valid points for homography")

    H, mask = cv2.findHomography(pixel_pts, world_pts, cv2.RANSAC)

    np.savez(args.out_h, H=H)

    print("Saved homography to:", args.out_h)

    # optional debug CSV
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["u", "v", "X_mm", "Y_mm"])
        for p, w in zip(pixel_pts, world_pts):
            writer.writerow([p[0], p[1], w[0], w[1]])

    print("Saved debug pairs to:", args.out_csv)


if __name__ == "__main__":
    main()